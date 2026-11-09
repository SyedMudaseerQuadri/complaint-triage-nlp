"""Shared paths, constants and the product-class mapping."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
ASSETS_DIR = ROOT / "assets"

SEED = 42
DATE_MIN = "2023-01-01"  # the CC0 mirror ends 2024-02-09, so the window is 2023-01-01 .. 2024-02-09
DATE_MAX = "2025-12-31"
MAX_PER_CLASS = 4000  # cap so the big classes (credit reporting) do not swamp the rest
MIN_WORDS = 20

# CFPB 'Product' value -> merged class (the team a complaint would be routed to).
# Near-duplicate / legacy names are merged; anything not listed is dropped.
PRODUCT_MAP = {
    "Credit reporting or other personal consumer reports": "credit_reporting",
    "Credit reporting, credit repair services, or other personal consumer reports": "credit_reporting",
    "Credit reporting": "credit_reporting",
    "Debt collection": "debt_collection",
    "Credit card": "credit_card",
    "Credit card or prepaid card": "credit_card",
    "Prepaid card": "credit_card",
    "Checking or savings account": "bank_account",
    "Bank account or service": "bank_account",
    "Mortgage": "mortgage",
    "Student loan": "student_loan",
    "Vehicle loan or lease": "vehicle_loan",
    "Payday loan, title loan, personal loan, or advance loan": "personal_loan",
    "Payday loan, title loan, or personal loan": "personal_loan",
    "Money transfer, virtual currency, or money service": "money_transfer",
    "Money transfers": "money_transfer",
}
LABELS = sorted(set(PRODUCT_MAP.values()))
