use std::{
    cell::Cell,
    fmt,
    sync::{
        atomic::{AtomicU32, Ordering},
        Condvar, Mutex,
    },
    time::{Duration, Instant},
};

use triomphe::Arc;

/// Fair, shared quota for pages pinned by in-flight write transactions.
///
/// The quota has a normal region and a reserved region. The normal region is
/// contested by all transactions. Young/small transactions can additionally use
/// the reserved region, while transactions that have already allocated more
/// than their small-transaction budget must spill dirty pages before taking
/// more normal quota. This lets a month-end batch keep making progress through
/// its existing spill path without being able to take the last page from small
/// foreground transactions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PageQuotaConfig {
    /// Total page units in the quota pool.
    pub capacity_pages: u32,
    /// Pages that only young/small transactions may use.
    pub reserved_pages: u32,
    /// Total allocations before a lease is treated as a large transaction.
    pub small_transaction_pages: u32,
    /// How long to wait for contested normal quota before reporting timeout.
    pub wait_timeout: Duration,
}

impl Default for PageQuotaConfig {
    fn default() -> Self {
        Self::new(256)
    }
}

impl PageQuotaConfig {
    /// Default wait used when an explicit configuration does not override it.
    pub const DEFAULT_WAIT_TIMEOUT: Duration = Duration::from_secs(30);

    /// Creates a configuration with sensible reserve and small-txn thresholds.
    pub fn new(capacity_pages: u32) -> Self {
        let capacity_pages = capacity_pages.max(4);
        let reserved_pages = (capacity_pages / 4).max(1).min(capacity_pages / 2);
        let small_transaction_pages = (capacity_pages / 8).clamp(8, 64).min(reserved_pages).max(1);
        Self {
            capacity_pages,
            reserved_pages,
            small_transaction_pages: small_transaction_pages.max(1),
            wait_timeout: Self::DEFAULT_WAIT_TIMEOUT,
        }
    }

    /// Derives a page quota from the shared page-cache byte budget.
    pub fn for_page_cache_size(page_cache_size: usize, page_size: u64) -> Self {
        let pages = (page_cache_size as u64 / page_size).max(16);
        let pages = u32::try_from(pages).unwrap_or(u32::MAX / 4);
        Self::new(pages)
    }

    /// Overrides the reserved page count.
    pub fn with_reserved_pages(mut self, pages: u32) -> Self {
        self.reserved_pages = pages.min(self.capacity_pages / 2).max(1);
        self.small_transaction_pages = self.small_transaction_pages.min(self.reserved_pages).max(1);
        self
    }

    /// Overrides the small-transaction threshold.
    pub fn with_small_transaction_pages(mut self, pages: u32) -> Self {
        self.small_transaction_pages = pages.max(1).min(self.reserved_pages);
        self
    }

    /// Overrides the wait timeout.
    pub fn with_wait_timeout(mut self, timeout: Duration) -> Self {
        self.wait_timeout = timeout;
        self
    }

    pub(crate) fn build(self) -> Arc<PageQuota> {
        Arc::new(PageQuota {
            normal_capacity: self.capacity_pages.saturating_sub(self.reserved_pages),
            config: self,
            state: Mutex::new(QuotaState::default()),
            wake: Condvar::new(),
            peak_used_pages: AtomicU32::new(0),
        })
    }
}

#[derive(Debug, Default)]
struct QuotaState {
    used_pages: u32,
    reserved_used_pages: u32,
}

/// A shared fair page quota.
pub struct PageQuota {
    config: PageQuotaConfig,
    normal_capacity: u32,
    state: Mutex<QuotaState>,
    wake: Condvar,
    peak_used_pages: AtomicU32,
}

impl fmt::Debug for PageQuota {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        f.debug_struct("PageQuota")
            .field("config", &self.config)
            .field("used_pages", &state.used_pages)
            .field("reserved_used_pages", &state.reserved_used_pages)
            .field(
                "peak_used_pages",
                &self.peak_used_pages.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl PageQuota {
    /// Creates a new quota pool.
    pub fn new(config: PageQuotaConfig) -> Arc<Self> {
        config.build()
    }

    /// Total configured pages.
    pub fn capacity_pages(&self) -> u32 {
        self.config.capacity_pages
    }

    /// Pages reserved for young/small transactions.
    pub fn reserved_pages(&self) -> u32 {
        self.config.reserved_pages
    }

    /// Current used pages, including reserved pages.
    pub fn used_pages(&self) -> u32 {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .used_pages
    }

    /// Current used reserved pages.
    pub fn reserved_used_pages(&self) -> u32 {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .reserved_used_pages
    }

    /// Largest total page usage observed by this quota pool.
    pub fn peak_used_pages(&self) -> u32 {
        self.peak_used_pages.load(Ordering::Relaxed)
    }

    pub(crate) fn wait_timeout(&self) -> Duration {
        self.config.wait_timeout
    }

    fn normal_free(&self, state: &QuotaState) -> u32 {
        let normal_used = state.used_pages.saturating_sub(state.reserved_used_pages);
        self.normal_capacity.saturating_sub(normal_used)
    }

    fn reserved_free(&self, state: &QuotaState) -> u32 {
        self.config
            .reserved_pages
            .saturating_sub(state.reserved_used_pages)
    }

    fn is_small(&self, lease: &PageQuotaLease) -> bool {
        lease.acquired_pages.get() < u64::from(self.config.small_transaction_pages)
    }

    fn can_wait_for(&self, state: &QuotaState, lease: &PageQuotaLease, pages: u32) -> bool {
        if pages == 0 || pages > self.normal_capacity {
            return true;
        }
        if self.normal_free(state) >= pages {
            return true;
        }
        self.is_small(lease)
            && self.reserved_free(state) >= pages
            && self.config.capacity_pages.saturating_sub(state.used_pages) >= pages
    }

    fn grant(&self, state: &mut QuotaState, lease: &PageQuotaLease, pages: u32, reserved: bool) {
        state.used_pages = state.used_pages.saturating_add(pages);
        if reserved {
            state.reserved_used_pages = state.reserved_used_pages.saturating_add(pages);
            lease
                .reserved_held_pages
                .set(lease.reserved_held_pages.get().saturating_add(pages));
        }
        lease
            .held_pages
            .set(lease.held_pages.get().saturating_add(pages));
        lease
            .acquired_pages
            .set(lease.acquired_pages.get().saturating_add(u64::from(pages)));
        self.peak_used_pages
            .fetch_max(state.used_pages, Ordering::Relaxed);
    }

    pub(crate) fn try_acquire(&self, lease: &PageQuotaLease, pages: u32) -> QuotaAdmission {
        if lease.bypass.get() || pages == 0 {
            return QuotaAdmission::Granted;
        }
        // A span larger than the normal share is an oversized value/allocation.
        // It cannot fit the fair bucket without deadlocking the caller, which is
        // already holding the contiguous page while building the node.
        if pages > self.normal_capacity {
            lease
                .acquired_pages
                .set(lease.acquired_pages.get().saturating_add(u64::from(pages)));
            return QuotaAdmission::Granted;
        }

        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if self.normal_free(&state) >= pages {
            self.grant(&mut state, lease, pages, false);
            QuotaAdmission::Granted
        } else if self.is_small(lease)
            && self.reserved_free(&state) >= pages
            && self.config.capacity_pages.saturating_sub(state.used_pages) >= pages
        {
            self.grant(&mut state, lease, pages, true);
            QuotaAdmission::Granted
        } else if lease.held_pages.get() > 0 {
            QuotaAdmission::Evict
        } else {
            QuotaAdmission::Wait
        }
    }

    /// Waits until the lease can acquire `pages` or the timeout expires.
    ///
    /// Returns `false` on timeout. Callers in a page construction path can use
    /// [`PageQuota::force_acquire`] when the current transaction has no
    /// evictable pages and waiting would self-deadlock on pinned nodes.
    pub(crate) fn wait_for_capacity(
        &self,
        lease: &PageQuotaLease,
        pages: u32,
        timeout: Duration,
    ) -> bool {
        if pages == 0 || lease.bypass.get() || pages > self.normal_capacity {
            return true;
        }
        let deadline = Instant::now()
            .checked_add(timeout)
            .unwrap_or_else(Instant::now);
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        while !self.can_wait_for(&state, lease, pages) {
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            let remaining = deadline - now;
            let (next, timeout_result) = self
                .wake
                .wait_timeout(state, remaining)
                .unwrap_or_else(|p| p.into_inner());
            state = next;
            if timeout_result.timed_out() && !self.can_wait_for(&state, lease, pages) {
                return false;
            }
        }
        true
    }

    /// Admits pages beyond the fair watermark to guarantee liveness for pinned
    /// recursion. The overshoot is still tracked and released normally.
    pub(crate) fn force_acquire(&self, lease: &PageQuotaLease, pages: u32) {
        if pages == 0 || lease.bypass.get() {
            return;
        }
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        self.grant(&mut state, lease, pages, false);
        self.wake.notify_all();
    }

    pub(crate) fn release(&self, lease: &PageQuotaLease, pages: u32) {
        if pages == 0 || lease.bypass.get() {
            return;
        }
        let pages = pages.min(lease.held_pages.get());
        if pages == 0 {
            return;
        }
        let reserved_pages = pages.min(lease.reserved_held_pages.get());
        {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            state.used_pages = state.used_pages.saturating_sub(pages);
            state.reserved_used_pages = state.reserved_used_pages.saturating_sub(reserved_pages);
        }
        lease.held_pages.set(lease.held_pages.get() - pages);
        lease
            .reserved_held_pages
            .set(lease.reserved_held_pages.get() - reserved_pages);
        self.wake.notify_all();
    }

    pub(crate) fn reconcile(&self, lease: &PageQuotaLease, actual_held_pages: u32) {
        let held = lease.held_pages.get();
        match actual_held_pages.cmp(&held) {
            std::cmp::Ordering::Less => self.release(lease, held - actual_held_pages),
            std::cmp::Ordering::Greater => self.force_acquire(lease, actual_held_pages - held),
            std::cmp::Ordering::Equal => {}
        }
    }

    fn release_lease(&self, lease: &PageQuotaLease) {
        let pages = lease.held_pages.get();
        if pages != 0 {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            state.used_pages = state.used_pages.saturating_sub(pages);
            state.reserved_used_pages = state
                .reserved_used_pages
                .saturating_sub(lease.reserved_held_pages.get());
            self.wake.notify_all();
        }
        lease.held_pages.set(0);
        lease.reserved_held_pages.set(0);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum QuotaAdmission {
    Granted,
    Evict,
    Wait,
}

/// Per-transaction quota accounting handle.
#[derive(Debug)]
pub struct PageQuotaLease {
    quota: Option<Arc<PageQuota>>,
    held_pages: Cell<u32>,
    reserved_held_pages: Cell<u32>,
    acquired_pages: Cell<u64>,
    bypass: Cell<bool>,
}

impl PageQuotaLease {
    pub(crate) fn new(quota: &Arc<PageQuota>) -> Self {
        Self {
            quota: Some(quota.clone()),
            held_pages: Cell::new(0),
            reserved_held_pages: Cell::new(0),
            acquired_pages: Cell::new(0),
            bypass: Cell::new(false),
        }
    }

    pub(crate) fn set_bypass(&self, bypass: bool) {
        self.bypass.set(bypass);
    }
}

impl Drop for PageQuotaLease {
    fn drop(&mut self) {
        if let Some(quota) = &self.quota {
            quota.release_lease(self);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_keeps_small_txn_available_when_large_holds_normal_pool() {
        let quota = PageQuotaConfig::new(32)
            .with_reserved_pages(8)
            .with_small_transaction_pages(4)
            .build();
        let large = PageQuotaLease::new(&quota);
        let small = PageQuotaLease::new(&quota);

        assert_eq!(quota.normal_capacity, 24);
        assert_eq!(quota.try_acquire(&large, 24), QuotaAdmission::Granted);
        assert!(large.acquired_pages.get() >= 24);
        assert_eq!(quota.used_pages(), 24);

        assert_eq!(quota.try_acquire(&small, 4), QuotaAdmission::Granted);
        assert_eq!(small.held_pages.get(), 4);
        assert_eq!(small.reserved_held_pages.get(), 4);
        assert_eq!(quota.reserved_used_pages(), 4);

        assert_eq!(quota.try_acquire(&large, 1), QuotaAdmission::Evict);
        quota.release(&large, 1);
        assert_eq!(quota.try_acquire(&large, 1), QuotaAdmission::Granted);
        assert_eq!(quota.used_pages(), 28);
    }

    #[test]
    fn exhausted_reserved_quota_times_out_instead_of_spinning() {
        let quota = PageQuotaConfig::new(8)
            .with_reserved_pages(2)
            .with_small_transaction_pages(2)
            .with_wait_timeout(Duration::from_millis(10))
            .build();
        let normal_holder = PageQuotaLease::new(&quota);
        let small_holder = PageQuotaLease::new(&quota);
        let waiting_small = PageQuotaLease::new(&quota);

        assert_eq!(
            quota.try_acquire(&normal_holder, 6),
            QuotaAdmission::Granted
        );
        assert_eq!(quota.try_acquire(&small_holder, 2), QuotaAdmission::Granted);
        assert_eq!(quota.try_acquire(&waiting_small, 1), QuotaAdmission::Wait);
        assert!(!quota.wait_for_capacity(&waiting_small, 1, Duration::from_millis(10)));

        quota.release(&small_holder, 1);
        assert!(quota.wait_for_capacity(&waiting_small, 1, Duration::from_secs(1)));
        assert_eq!(
            quota.try_acquire(&waiting_small, 1),
            QuotaAdmission::Granted
        );
    }
}
