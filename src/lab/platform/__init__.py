"""Local persistence, data access, and orchestration adapters."""

from lab.platform.campaign_store import CampaignStore
from lab.platform.evidence_store import EvidenceStore
from lab.platform.market_store import ProviderBatchStore, SnapshotCatalog
from lab.platform.report_store import ReportStore

__all__ = ["CampaignStore", "EvidenceStore", "ProviderBatchStore", "ReportStore", "SnapshotCatalog"]
