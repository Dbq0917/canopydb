use std::{
    collections::HashSet,
    mem::size_of,
    panic::{catch_unwind, AssertUnwindSafe},
};

use zerocopy::FromBytes;

use crate::{
    allocator::MainAllocator,
    error::{io_invalid_data, Error},
    node::{LeafNode, UntypedNode},
    page::Page,
    repr::{
        header_cast, CompressedPageHeader, IndirectionValue, MaybeValue, PageFlags, PageHeader,
        PageId, PageTrait, TreeValue, FIRST_COMPRESSED_PAGE,
    },
    utils::EscapedBytes,
    ReservedPageHeader, SharedDatabaseInner, Transaction, PAGE_SIZE,
};

/// Statistics produced by a successful online integrity check.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct IntegrityReport {
    /// Number of user trees checked.
    pub trees_checked: u64,
    /// Number of tree nodes (pages) checked.
    pub pages_checked: u64,
    /// Number of key/value pairs checked in tree leaves.
    pub keys_checked: u64,
    /// Number of overflow value pages checked.
    pub overflow_pages_checked: u64,
}

impl IntegrityReport {
    /// A successful check means the database was consistent, so this always returns `true`.
    #[inline]
    pub fn is_ok(&self) -> bool {
        true
    }
}

#[derive(Debug, Default)]
struct ScanStats {
    trees: u64,
    pages: u64,
    user_keys: u64,
    overflow_pages: u64,
}

#[derive(Debug, Clone, Copy)]
struct PageLimits {
    normal: PageId,
    compressed: PageId,
}

struct PageTracker {
    limits: PageLimits,
    normal_seen: Vec<u64>,
    compressed_seen: Vec<u64>,
}

impl PageTracker {
    fn new(limits: PageLimits) -> Self {
        Self {
            limits,
            normal_seen: Default::default(),
            compressed_seen: Default::default(),
        }
    }

    fn set_bit(bits: &mut Vec<u64>, index: u32, what: &str) -> Result<(), Error> {
        let word = index as usize / 64;
        if bits.len() <= word {
            bits.resize(word + 1, 0);
        }
        let mask = 1u64 << (index % 64);
        if bits[word] & mask != 0 {
            return Err(io_invalid_data!(
                "Page {index} is referenced more than once ({what})"
            ));
        }
        bits[word] |= mask;
        Ok(())
    }

    fn mark_normal(&mut self, page_id: PageId, span: PageId) -> Result<(), Error> {
        if page_id < 2 {
            return Err(io_invalid_data!(
                "Page {page_id} is in the reserved page range"
            ));
        }
        if span == 0 {
            return Err(io_invalid_data!("Page {page_id} has a zero span"));
        }
        let end = page_id
            .checked_add(span)
            .ok_or_else(|| io_invalid_data!("Page {page_id} span overflows"))?;
        if end > self.limits.normal {
            return Err(io_invalid_data!(
                "Page {page_id} span {span} extends past the allocation horizon {}",
                self.limits.normal
            ));
        }
        for id in page_id..end {
            Self::set_bit(&mut self.normal_seen, id, "normal page")?;
        }
        Ok(())
    }

    fn mark_compressed(&mut self, page_id: PageId) -> Result<(), Error> {
        if page_id < FIRST_COMPRESSED_PAGE || page_id >= self.limits.compressed {
            return Err(io_invalid_data!(
                "Compressed page {page_id} is outside the allocation horizon {}..{}",
                FIRST_COMPRESSED_PAGE,
                self.limits.compressed
            ));
        }
        Self::set_bit(
            &mut self.compressed_seen,
            page_id - FIRST_COMPRESSED_PAGE,
            "compressed page",
        )
    }
}

#[derive(Debug)]
struct NodeRange {
    min: Vec<u8>,
    max: Vec<u8>,
    keys: u64,
}

struct CheckContext<'a> {
    txn: &'a Transaction,
    limits: PageLimits,
    tracker: &'a mut PageTracker,
    stats: &'a mut ScanStats,
}

fn corrupt(msg: impl Into<String>) -> Error {
    io_invalid_data!("{}", msg.into())
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

pub(crate) fn check_integrity(inner: &SharedDatabaseInner) -> Result<IntegrityReport, Error> {
    catch_check(|| {
        let txn = Transaction::new_read(inner, true)?;
        check_transaction(&txn)
    })
}

pub(crate) fn check_read_transaction(txn: &Transaction) -> Result<IntegrityReport, Error> {
    catch_check(|| check_transaction(txn))
}

fn catch_check(
    f: impl FnOnce() -> Result<IntegrityReport, Error>,
) -> Result<IntegrityReport, Error> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => Err(io_invalid_data!(
            "Database integrity check panicked: {}",
            panic_message(payload.as_ref())
        )),
    }
}

fn check_transaction(txn: &Transaction) -> Result<IntegrityReport, Error> {
    txn.state.get().check_halted()?;

    let limits = {
        let allocator = txn.inner.allocator.lock();
        PageLimits {
            normal: allocator.next_page_id,
            compressed: allocator.next_indirection_id,
        }
    };
    if limits.normal < 2 || limits.normal > FIRST_COMPRESSED_PAGE {
        return Err(corrupt(format!(
            "Invalid normal page allocation horizon {}",
            limits.normal
        )));
    }
    if limits.compressed < FIRST_COMPRESSED_PAGE {
        return Err(corrupt(format!(
            "Compressed page allocation horizon {} is below the compressed page range",
            limits.compressed
        )));
    }

    let mut tracker = PageTracker::new(limits);
    let mut stats = ScanStats::default();
    let metapage = txn.state.get().metapage;
    let file_len = txn.inner.file.file_len();
    if file_len < 2 * PAGE_SIZE || file_len % PAGE_SIZE != 0 {
        return Err(corrupt(format!(
            "Database file has an invalid length of {file_len} bytes"
        )));
    }

    check_metapages(txn)?;

    // The freelist is stored outside of the user trees, so account for its pages too.
    if metapage.freelist_root != 0 {
        let (spans, data) = txn
            .inner
            .read_from_pages(metapage.freelist_root, true)
            .map_err(|e| corrupt(format!("Unable to read the persistent freelist: {e}")))?;
        for (page_id, span) in spans.iter_spans() {
            tracker.mark_normal(page_id, span)?;
        }
        let allocator = MainAllocator::from_bytes(&data)
            .map_err(|e| corrupt(format!("Invalid persistent freelist: {e}")))?;
        if allocator.next_page_id < 2 || allocator.next_page_id > FIRST_COMPRESSED_PAGE {
            return Err(corrupt(format!(
                "Persistent freelist has an invalid normal page horizon {}",
                allocator.next_page_id
            )));
        }
        if allocator.next_indirection_id < FIRST_COMPRESSED_PAGE {
            return Err(corrupt(format!(
                "Persistent freelist has an invalid indirection horizon {}",
                allocator.next_indirection_id
            )));
        }
        let required_file_len = allocator.next_page_id as u64 * PAGE_SIZE;
        if file_len < required_file_len {
            return Err(corrupt(format!(
                "Database file is {file_len} bytes but the persistent allocator requires {required_file_len}"
            )));
        }
    }

    // The metapage's tree catalog is itself a tree.
    let trees_value = metapage.trees_tree;
    let declared_catalog_keys = trees_value.num_keys;
    let catalog = {
        let mut ctx = CheckContext {
            txn,
            limits,
            tracker: &mut tracker,
            stats: &mut stats,
        };
        check_tree(&mut ctx, &trees_value, "tree catalog")?
    };
    if catalog != declared_catalog_keys {
        return Err(corrupt(format!(
            "Tree catalog declares {} keys but contains {catalog}",
            declared_catalog_keys
        )));
    }

    let mut tree_entries = Vec::new();
    for entry in txn
        .get_trees_tree()
        .iter()
        .map_err(|e| corrupt(format!("Unable to iterate the tree catalog: {e}")))?
    {
        let (name, value) =
            entry.map_err(|e| corrupt(format!("Unable to read the tree catalog: {e}")))?;
        if value.len() != size_of::<TreeValue>() {
            return Err(corrupt(format!(
                "Tree {} has a metadata value of {} bytes, expected {}",
                EscapedBytes(&name),
                value.len(),
                size_of::<TreeValue>()
            )));
        }
        let tree_value = *TreeValue::ref_from_bytes(value.as_ref()).map_err(|_| {
            corrupt(format!(
                "Tree {} has an invalid metadata value",
                EscapedBytes(&name)
            ))
        })?;
        validate_tree_value(&name, &tree_value)?;
        tree_entries.push((name, tree_value));
    }
    if tree_entries.len() as u64 != catalog {
        return Err(corrupt(format!(
            "Tree catalog changed while checking: expected {catalog} entries, read {}",
            tree_entries.len()
        )));
    }

    let mut tree_ids = HashSet::new();
    for (name, tree_value) in &tree_entries {
        let tree_id = tree_value.id;
        if !tree_ids.insert(tree_id) {
            return Err(corrupt(format!(
                "Tree {} reuses tree id {}",
                EscapedBytes(name),
                tree_id
            )));
        }
        let keys = {
            let mut ctx = CheckContext {
                txn,
                limits,
                tracker: &mut tracker,
                stats: &mut stats,
            };
            check_tree(
                &mut ctx,
                tree_value,
                &format!("tree {}", EscapedBytes(name)),
            )?
        };
        let declared_keys = tree_value.num_keys;
        if keys != declared_keys {
            return Err(corrupt(format!(
                "Tree {} declares {} keys but contains {keys}",
                EscapedBytes(name),
                declared_keys
            )));
        }
        stats.trees += 1;
        stats.user_keys = stats
            .user_keys
            .checked_add(keys)
            .ok_or_else(|| corrupt("Key count overflow"))?;
    }

    // The indirection map is a tree as well. Validate its entries and the physical pages they
    // point to. This also covers compressed pages that are no longer reachable from a user tree
    // but are still part of the current snapshot.
    let indirections_value = metapage.indirections_tree;
    let indirections = {
        let mut ctx = CheckContext {
            txn,
            limits,
            tracker: &mut tracker,
            stats: &mut stats,
        };
        check_tree(&mut ctx, &indirections_value, "indirection catalog")?
    };
    let declared_indirections = indirections_value.num_keys;
    if indirections != declared_indirections {
        return Err(corrupt(format!(
            "Indirection catalog declares {} keys but contains {indirections}",
            declared_indirections
        )));
    }

    let mut targets = PageTracker::new(limits);
    for entry in txn
        .get_indirection_tree()
        .iter()
        .map_err(|e| corrupt(format!("Unable to iterate the indirection catalog: {e}")))?
    {
        let (key, value) =
            entry.map_err(|e| corrupt(format!("Unable to read the indirection catalog: {e}")))?;
        if key.len() != size_of::<PageId>() {
            return Err(corrupt(format!(
                "Indirection key has {} bytes, expected {}",
                key.len(),
                size_of::<PageId>()
            )));
        }
        let logical_page = PageId::from_be_bytes(
            key.as_ref()
                .try_into()
                .map_err(|_| corrupt("Invalid indirection key"))?,
        );
        if logical_page < FIRST_COMPRESSED_PAGE || logical_page >= limits.compressed {
            return Err(corrupt(format!(
                "Indirection key {logical_page} is outside the compressed page range"
            )));
        }
        if value.len() != size_of::<IndirectionValue>() {
            return Err(corrupt(format!(
                "Indirection value for page {logical_page} has {} bytes, expected {}",
                value.len(),
                size_of::<IndirectionValue>()
            )));
        }
        let indirection = IndirectionValue::ref_from_bytes(value.as_ref())
            .map_err(|_| corrupt(format!("Invalid indirection value for page {logical_page}")))?;
        let physical_page = indirection.pid;
        let span = PageId::from(indirection.span);
        targets.mark_normal(physical_page, span)?;

        let page = txn
            .inner
            .read_page_at(physical_page, physical_page, Some(span))
            .map_err(|e| {
                corrupt(format!(
                    "Unable to read compressed page {logical_page} at {physical_page}: {e}"
                ))
            })?;
        verify_compressed_page(&page, physical_page, span, logical_page)?;
    }

    Ok(IntegrityReport {
        trees_checked: stats.trees,
        pages_checked: stats.pages,
        keys_checked: stats.user_keys,
        overflow_pages_checked: stats.overflow_pages,
    })
}

fn check_metapages(txn: &Transaction) -> Result<(), Error> {
    // Both metapages are checked from disk, even when the in-memory state is newer. This
    // detects bit rot in the pages used to open the database after a restart.
    // If a checkpoint or compaction is active, skip this small disk-only check instead of
    // waiting. Waiting here could deadlock with compaction (which holds this lock while waiting
    // for the read transaction doing the integrity check), and an active checkpoint may be in
    // the middle of rewriting the inactive metapage.
    let Some(_checkpoint_lock) = txn.inner.checkpoint_lock.try_lock() else {
        return Ok(());
    };
    for page_id in 0..=1 {
        let disk_metapage = txn
            .inner
            .read_metapage(page_id)
            .map_err(|e| corrupt(format!("Invalid metapage {page_id}: {e}")))?;
        if disk_metapage.tx_id < disk_metapage.snapshot_tx_id {
            return Err(corrupt(format!(
                "Metapage {page_id} has a snapshot newer than its transaction"
            )));
        }
        if disk_metapage.wal_start > disk_metapage.wal_end {
            return Err(corrupt(format!(
                "Metapage {page_id} has an invalid WAL range {}..{}",
                disk_metapage.wal_start, disk_metapage.wal_end
            )));
        }
    }
    Ok(())
}

fn validate_tree_value(name: &[u8], value: &TreeValue) -> Result<(), Error> {
    let root = value.root;
    let num_keys = value.num_keys;
    let level = value.level;
    let fixed_key_len = value.fixed_key_len;
    let fixed_value_len = value.fixed_value_len;

    if (fixed_key_len < 0) != (fixed_value_len < 0) {
        return Err(corrupt(format!(
            "Tree {} mixes fixed and variable length keys/values",
            EscapedBytes(name)
        )));
    }
    if root == 0 {
        if num_keys != 0 || level != 0 {
            return Err(corrupt(format!(
                "Empty tree {} has keys {} and level {}",
                EscapedBytes(name),
                num_keys,
                level
            )));
        }
    } else if root < 2 {
        return Err(corrupt(format!(
            "Tree {} has an invalid root page {root}",
            EscapedBytes(name)
        )));
    } else if num_keys == 0 {
        return Err(corrupt(format!(
            "Tree {} has a root but no keys",
            EscapedBytes(name)
        )));
    }
    if value.min_branch_node_pages == 0 || value.min_leaf_node_pages == 0 {
        return Err(corrupt(format!(
            "Tree {} has a zero minimum node span",
            EscapedBytes(name)
        )));
    }
    Ok(())
}

fn check_tree(ctx: &mut CheckContext<'_>, value: &TreeValue, label: &str) -> Result<u64, Error> {
    if value.root == 0 {
        return Ok(0);
    }
    let range = walk_tree_node(ctx, value, value.root, value.level, label)?;
    Ok(range.keys)
}

fn walk_tree_node(
    ctx: &mut CheckContext<'_>,
    tree: &TreeValue,
    page_id: PageId,
    expected_level: u8,
    label: &str,
) -> Result<NodeRange, Error> {
    let node = load_node(ctx, page_id, label)?;
    let header = *node.node_header();
    if header.level != expected_level {
        return Err(corrupt(format!(
            "{label}: page {page_id} is at level {}, expected {expected_level}",
            header.level
        )));
    }
    if header.fixed_key_len != tree.fixed_key_len || header.fixed_value_len != tree.fixed_value_len
    {
        return Err(corrupt(format!(
            "{label}: page {page_id} has fixed key/value lengths {}/{} but the tree declares {}/{}",
            header.fixed_key_len, header.fixed_value_len, tree.fixed_key_len, tree.fixed_value_len
        )));
    }

    if header.level == 0 {
        return check_leaf(ctx, tree, &node, label);
    }
    check_branch(ctx, tree, &node, expected_level, label)
}

fn check_leaf(
    ctx: &mut CheckContext<'_>,
    tree: &TreeValue,
    node: &UntypedNode,
    label: &str,
) -> Result<NodeRange, Error> {
    let leaf = node.as_leaf();
    let num_keys = leaf.num_keys();
    let mut prev: Option<Vec<u8>> = None;
    let mut min = None;
    let mut max = None;
    for i in 0..num_keys {
        let key = leaf.full_key_at(i).into_owned();
        if prev.as_ref().is_some_and(|prev| prev >= &key) {
            return Err(corrupt(format!(
                "{label}: keys in leaf {} are not strictly increasing at index {i}",
                node.id()
            )));
        }
        if min.is_none() {
            min = Some(key.clone());
        }
        max = Some(key.clone());
        prev = Some(key);

        match leaf.value_at(i) {
            MaybeValue::Bytes(value) => {
                if tree.fixed_value_len >= 0 && value.len() != tree.fixed_value_len as usize {
                    return Err(corrupt(format!(
                        "{label}: value at leaf {} index {i} has {} bytes, expected {}",
                        node.id(),
                        value.len(),
                        tree.fixed_value_len
                    )));
                }
            }
            MaybeValue::Overflow([overflow_page, span]) => {
                if tree.fixed_value_len >= 0 {
                    return Err(corrupt(format!(
                        "{label}: leaf {} index {i} uses overflow storage in a fixed length tree",
                        node.id()
                    )));
                }
                check_overflow(ctx, overflow_page, span, label)?;
            }
            MaybeValue::Delete => {
                return Err(corrupt(format!(
                    "{label}: leaf {} index {i} contains a tombstone",
                    node.id()
                )));
            }
        }
    }
    Ok(NodeRange {
        min: min.ok_or_else(|| corrupt(format!("{label}: leaf {} is empty", node.id())))?,
        max: max.ok_or_else(|| corrupt(format!("{label}: leaf {} is empty", node.id())))?,
        keys: num_keys as u64,
    })
}

fn check_branch(
    ctx: &mut CheckContext<'_>,
    tree: &TreeValue,
    node: &UntypedNode,
    expected_level: u8,
    label: &str,
) -> Result<NodeRange, Error> {
    let branch = node.as_branch();
    let num_keys = branch.num_keys();
    let mut separators = Vec::with_capacity(num_keys);
    for i in 0..num_keys {
        let separator = branch.full_key_at(i).into_owned();
        if separators
            .last()
            .is_some_and(|prev: &Vec<u8>| prev >= &separator)
        {
            return Err(corrupt(format!(
                "{label}: separators in branch {} are not strictly increasing at index {i}",
                node.id()
            )));
        }
        separators.push(separator);
    }

    let child_level = expected_level.checked_sub(1).ok_or_else(|| {
        corrupt(format!(
            "{label}: branch {} is at an invalid level",
            node.id()
        ))
    })?;
    let mut min = None;
    let mut max = None;
    let mut keys = 0u64;
    for child_idx in 0..=num_keys {
        let child_page = branch.pointer_at(child_idx);
        if child_page == 0 {
            return Err(corrupt(format!(
                "{label}: branch {} has a zero child pointer at index {child_idx}",
                node.id()
            )));
        }
        let child = walk_tree_node(ctx, tree, child_page, child_level, label)?;
        if child_idx > 0 && child.min.as_slice() < separators[child_idx - 1].as_slice() {
            return Err(corrupt(format!(
                "{label}: branch {} child {child_idx} starts before its separator",
                node.id()
            )));
        }
        if child_idx < num_keys && child.max.as_slice() >= separators[child_idx].as_slice() {
            return Err(corrupt(format!(
                "{label}: branch {} child {child_idx} ends at or after its separator",
                node.id()
            )));
        }
        if min.is_none() {
            min = Some(child.min);
        }
        max = Some(child.max);
        keys = keys
            .checked_add(child.keys)
            .ok_or_else(|| corrupt(format!("{label}: key count overflow")))?;
    }

    Ok(NodeRange {
        min: min
            .ok_or_else(|| corrupt(format!("{label}: branch {} has no children", node.id())))?,
        max: max
            .ok_or_else(|| corrupt(format!("{label}: branch {} has no children", node.id())))?,
        keys,
    })
}

fn check_overflow(
    ctx: &mut CheckContext<'_>,
    page_id: PageId,
    expected_span: PageId,
    label: &str,
) -> Result<(), Error> {
    let node = load_node(ctx, page_id, label)?;
    let header = *node.node_header();
    if header.level != 0 {
        return Err(corrupt(format!(
            "{label}: overflow page {page_id} is not a leaf"
        )));
    }
    if header.fixed_key_len != -1 || header.fixed_value_len != -1 {
        return Err(corrupt(format!(
            "{label}: overflow page {page_id} has fixed key/value lengths"
        )));
    }
    if node.span() != expected_span {
        return Err(corrupt(format!(
            "{label}: overflow page {page_id} has span {}, expected {expected_span}",
            node.span()
        )));
    }
    let leaf: &LeafNode = node.as_leaf();
    if leaf.num_keys() != 1 {
        return Err(corrupt(format!(
            "{label}: overflow page {page_id} has {} keys, expected 1",
            leaf.num_keys()
        )));
    }
    if !leaf.full_key_at(0).is_empty() {
        return Err(corrupt(format!(
            "{label}: overflow page {page_id} has a non-empty key"
        )));
    }
    match leaf.value_at(0) {
        MaybeValue::Bytes(_) => (),
        MaybeValue::Overflow(_) => {
            return Err(corrupt(format!(
                "{label}: overflow page {page_id} recursively points to overflow storage"
            )));
        }
        MaybeValue::Delete => {
            return Err(corrupt(format!(
                "{label}: overflow page {page_id} contains a tombstone"
            )));
        }
    }
    ctx.stats.overflow_pages += 1;
    Ok(())
}

fn load_node(
    ctx: &mut CheckContext<'_>,
    page_id: PageId,
    label: &str,
) -> Result<UntypedNode, Error> {
    if page_id < 2 {
        return Err(corrupt(format!("{label}: invalid page id {page_id}")));
    }
    if page_id < FIRST_COMPRESSED_PAGE {
        if page_id >= ctx.limits.normal {
            return Err(corrupt(format!(
                "{label}: page {page_id} is beyond the allocation horizon {}",
                ctx.limits.normal
            )));
        }
    } else if page_id >= ctx.limits.compressed {
        return Err(corrupt(format!(
            "{label}: compressed page {page_id} is beyond the allocation horizon {}",
            ctx.limits.compressed
        )));
    }

    let node = ctx
        .txn
        .read_clean_node_uncached(page_id)
        .map_err(|e| corrupt(format!("{label}: unable to read page {page_id}: {e}")))?;
    node.validate_layout()
        .map_err(|e| corrupt(format!("{label}: invalid page {page_id}: {e}")))?;
    let header = *node.node_header();
    if header.page_header.id != page_id {
        return Err(corrupt(format!(
            "{label}: page {page_id} contains a header for page {}",
            header.page_header.id
        )));
    }
    let actual_span =
        ((size_of::<ReservedPageHeader>() + node.page_size()) / PAGE_SIZE as usize) as PageId;
    if actual_span == 0 || u32::from(header.page_header.span) != actual_span {
        return Err(corrupt(format!(
            "{label}: page {page_id} declares span {} but occupies {actual_span} pages",
            u32::from(header.page_header.span)
        )));
    }

    if page_id.is_compressed() {
        ctx.tracker.mark_compressed(page_id)?;
        if let Some((physical_page, physical_span)) = node.compressed_page {
            ctx.tracker.mark_normal(physical_page, physical_span)?;
        }
    } else {
        ctx.tracker.mark_normal(page_id, actual_span)?;
    }
    ctx.stats.pages += 1;
    Ok(node)
}

fn verify_compressed_page(
    page: &Page,
    physical_page: PageId,
    span: PageId,
    logical_page: PageId,
) -> Result<(), Error> {
    let header: &PageHeader = header_cast(page);
    if header.id != physical_page {
        return Err(corrupt(format!(
            "Compressed page {logical_page} points to page {physical_page} but the page header has id {}",
            header.id
        )));
    }
    if u32::from(header.span) != span {
        return Err(corrupt(format!(
            "Compressed page {logical_page} points to span {span} but the page header has span {}",
            u32::from(header.span)
        )));
    }
    if page.check_checksum() == Some(false) {
        return Err(corrupt(format!(
            "Checksum mismatch for compressed page {logical_page} at {physical_page}"
        )));
    }
    if header.flags.contains(PageFlags::Compressed) {
        let compressed: &CompressedPageHeader = header_cast(page);
        let available = page
            .data()
            .len()
            .checked_sub(size_of::<CompressedPageHeader>())
            .ok_or_else(|| corrupt(format!("Compressed page {physical_page} is too small")))?;
        if compressed.compressed_len as usize > available {
            return Err(corrupt(format!(
                "Compressed page {physical_page} declares {} bytes but only {available} are available",
                compressed.compressed_len
            )));
        }
        if compressed.compressed_len == 0 {
            return Err(corrupt(format!(
                "Compressed page {physical_page} has an empty compressed payload"
            )));
        }
        let uncompressed_len = compressed.uncompressed_len as usize;
        if uncompressed_len == 0 {
            return Err(corrupt(format!(
                "Compressed page {physical_page} has an invalid uncompressed length {uncompressed_len}"
            )));
        }
    }
    Ok(())
}
