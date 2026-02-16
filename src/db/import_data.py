"""Data import scripts for Med-I-C structured documents."""

import pandas as pd
import re
from pathlib import Path
from .database import (
    get_connection, init_database, execute_many,
    DOCS_DIR, DB_PATH
)


def safe_float(value):
    """Safely convert a value to float, returning None on failure."""
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def safe_int(value):
    """Safely convert a value to int, returning None on failure."""
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def classify_severity(description: str) -> str:
    """Classify drug interaction severity based on description keywords."""
    if not description:
        return "unknown"

    desc_lower = description.lower()

    # Major severity indicators
    major_keywords = [
        "cardiotoxic", "nephrotoxic", "hepatotoxic", "neurotoxic",
        "fatal", "death", "severe", "contraindicated", "arrhythmia",
        "qt prolongation", "seizure", "bleeding", "hemorrhage",
        "serotonin syndrome", "neuroleptic malignant"
    ]

    # Moderate severity indicators
    moderate_keywords = [
        "increase", "decrease", "reduce", "enhance", "inhibit",
        "metabolism", "concentration", "absorption", "excretion",
        "therapeutic effect", "adverse effect", "toxicity"
    ]

    for keyword in major_keywords:
        if keyword in desc_lower:
            return "major"

    for keyword in moderate_keywords:
        if keyword in desc_lower:
            return "moderate"

    return "minor"


def import_eml_antibiotics() -> int:
    """Import WHO EML antibiotic classification data."""
    print("Importing EML antibiotic data...")

    eml_files = {
        "ACCESS": DOCS_DIR / "antibiotic_guidelines" / "EML export-ACCESS group.xlsx",
        "RESERVE": DOCS_DIR / "antibiotic_guidelines" / "EML export-RESERVE group.xlsx",
        "WATCH": DOCS_DIR / "antibiotic_guidelines" / "EML export-WATCH group.xlsx",
    }

    records = []
    for category, filepath in eml_files.items():
        if not filepath.exists():
            print(f"  Warning: {filepath} not found, skipping...")
            continue

        try:
            # Use openpyxl directly with read_only=True for faster loading
            import openpyxl
            wb = openpyxl.load_workbook(filepath, read_only=True)
            ws = wb.active

            # Get headers from first row
            headers = []
            for cell in ws[1]:
                headers.append(str(cell.value).strip().lower().replace(' ', '_') if cell.value else f'col_{len(headers)}')

            # Process data rows
            for row_idx, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
                row_dict = dict(zip(headers, row))

                medicine = str(row_dict.get('medicine_name', row_dict.get('medicine', '')))
                if not medicine or medicine == 'None' or medicine == 'nan':
                    continue

                def safe_str(val):
                    if val is None or pd.isna(val):
                        return ''
                    return str(val)

                records.append((
                    medicine,
                    category,
                    safe_str(row_dict.get('eml_section', '')),
                    safe_str(row_dict.get('formulations', '')),
                    safe_str(row_dict.get('indication', '')),
                    safe_str(row_dict.get('atc_codes', row_dict.get('atc_code', ''))),
                    safe_str(row_dict.get('combined_with', '')),
                    safe_str(row_dict.get('status', '')),
                ))

            wb.close()
            print(f"  Loaded {len([r for r in records if r[1] == category])} from {category}")

        except Exception as e:
            print(f"  Warning: Error reading {filepath}: {e}")
            continue

    if records:
        query = """
            INSERT INTO eml_antibiotics
            (medicine_name, who_category, eml_section, formulations,
             indication, atc_codes, combined_with, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        execute_many(query, records)
        print(f"  Imported {len(records)} EML antibiotic records total")

    return len(records)


def import_atlas_susceptibility() -> int:
    """Import ATLAS antimicrobial susceptibility data."""
    print("Importing ATLAS susceptibility data...")

    filepath = DOCS_DIR / "pathogen_resistance" / "ATLAS Susceptibility Data Export.xlsx"

    if not filepath.exists():
        print(f"  Warning: {filepath} not found, skipping...")
        return 0

    # Read the raw data to find the header row and extract region
    df_raw = pd.read_excel(filepath, sheet_name="Percent", header=None)

    # Extract region from the title (row 1)
    region = "Unknown"
    for idx, row in df_raw.head(5).iterrows():
        cell = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
        if "from" in cell.lower():
            # Extract country from "Percentage Susceptibility from Argentina"
            parts = cell.split("from")
            if len(parts) > 1:
                region = parts[1].strip()
            break

    # Find the header row (contains 'Antibacterial' or 'N')
    header_row = 4  # Default
    for idx, row in df_raw.head(10).iterrows():
        if any('Antibacterial' in str(v) for v in row.values if pd.notna(v)):
            header_row = idx
            break

    # Read with proper header
    df = pd.read_excel(filepath, sheet_name="Percent", header=header_row)

    # Standardize column names
    df.columns = [str(col).strip().lower().replace(' ', '_').replace('.', '') for col in df.columns]

    records = []
    for _, row in df.iterrows():
        antibiotic = str(row.get('antibacterial', ''))

        # Skip empty or non-antibiotic rows
        if not antibiotic or antibiotic == 'nan' or 'omitted' in antibiotic.lower():
            continue
        if 'in vitro' in antibiotic.lower() or 'table cells' in antibiotic.lower():
            continue

        # Get susceptibility values
        n_value = row.get('n', None)
        pct_s = row.get('susc', row.get('susceptible', None))
        pct_i = row.get('int', row.get('intermediate', None))
        pct_r = row.get('res', row.get('resistant', None))

        # Use safe conversion functions
        n_int = safe_int(n_value)
        s_float = safe_float(pct_s)

        if n_int is not None and s_float is not None:
            records.append((
                "General",  # Species - will be refined if more data available
                "",  # Family
                antibiotic,
                s_float,
                safe_float(pct_i),
                safe_float(pct_r),
                n_int,
                2024,  # Year - from the data context
                region,
                "ATLAS"
            ))

    if records:
        query = """
            INSERT INTO atlas_susceptibility
            (species, family, antibiotic, percent_susceptible,
             percent_intermediate, percent_resistant, total_isolates,
             year, region, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        execute_many(query, records)
        print(f"  Imported {len(records)} ATLAS susceptibility records from {region}")

    return len(records)


def import_mic_breakpoints() -> int:
    """Import EUCAST MIC breakpoint tables."""
    print("Importing MIC breakpoint data...")

    filepath = DOCS_DIR / "mic_breakpoints" / "v_16.0__BreakpointTables.xlsx"

    if not filepath.exists():
        print(f"  Warning: {filepath} not found, skipping...")
        return 0

    # Get all sheet names
    xl = pd.ExcelFile(filepath)

    # Skip non-pathogen sheets
    skip_sheets = {'Content', 'Changes', 'Notes', 'Guidance', 'Dosages',
                   'Technical uncertainty', 'PK PD breakpoints', 'PK PD cutoffs'}

    records = []
    for sheet_name in xl.sheet_names:
        if sheet_name in skip_sheets:
            continue

        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name, header=None)

            # Try to find antibiotic data - look for rows with MIC values
            pathogen_group = sheet_name

            # Simple heuristic: look for rows that might contain antibiotic names and MIC values
            for idx, row in df.iterrows():
                row_values = [str(v).strip() for v in row.values if pd.notna(v)]

                # Look for rows that might be antibiotic entries
                if len(row_values) >= 2:
                    potential_antibiotic = row_values[0]

                    # Skip header-like rows
                    if any(kw in potential_antibiotic.lower() for kw in
                           ['antibiotic', 'agent', 'note', 'disk', 'mic', 'breakpoint']):
                        continue

                    # Try to extract MIC values (numbers)
                    mic_values = []
                    for v in row_values[1:]:
                        try:
                            mic_values.append(float(v.replace('≤', '').replace('>', '').replace('<', '').strip()))
                        except (ValueError, AttributeError):
                            pass

                    if len(mic_values) >= 2 and len(potential_antibiotic) > 2:
                        records.append((
                            pathogen_group,
                            potential_antibiotic,
                            None,  # route
                            mic_values[0] if len(mic_values) > 0 else None,  # S breakpoint
                            mic_values[1] if len(mic_values) > 1 else None,  # R breakpoint
                            None,  # disk S
                            None,  # disk R
                            None,  # notes
                            "16.0"
                        ))
        except Exception as e:
            print(f"  Warning: Could not parse sheet '{sheet_name}': {e}")
            continue

    if records:
        query = """
            INSERT INTO mic_breakpoints
            (pathogen_group, antibiotic, route, mic_susceptible, mic_resistant,
             disk_susceptible, disk_resistant, notes, eucast_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        execute_many(query, records)
        print(f"  Imported {len(records)} MIC breakpoint records")

    return len(records)


def import_drug_interactions(limit: int = None) -> int:
    """Import drug-drug interaction database."""
    print("Importing drug interactions data...")

    filepath = DOCS_DIR / "drug_safety" / "db_drug_interactions.csv"

    if not filepath.exists():
        print(f"  Warning: {filepath} not found, skipping...")
        return 0

    # Read CSV in chunks due to large size
    chunk_size = 10000
    total_records = 0

    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        # Standardize column names
        chunk.columns = [col.strip().lower().replace(' ', '_') for col in chunk.columns]

        records = []
        for _, row in chunk.iterrows():
            drug_1 = str(row.get('drug_1', row.get('drug1', row.iloc[0] if len(row) > 0 else '')))
            drug_2 = str(row.get('drug_2', row.get('drug2', row.iloc[1] if len(row) > 1 else '')))
            description = str(row.get('interaction_description', row.get('description',
                             row.get('interaction', row.iloc[2] if len(row) > 2 else ''))))

            severity = classify_severity(description)

            if drug_1 and drug_2:
                records.append((drug_1, drug_2, description, severity))

        if records:
            query = """
                INSERT INTO drug_interactions
                (drug_1, drug_2, interaction_description, severity)
                VALUES (?, ?, ?, ?)
            """
            execute_many(query, records)
            total_records += len(records)

        if limit and total_records >= limit:
            break

    print(f"  Imported {total_records} drug interaction records")
    return total_records


def import_all_data(interactions_limit: int = None) -> dict:
    """Import all structured data into the database."""
    print(f"\n{'='*50}")
    print("Med-I-C Data Import")
    print(f"{'='*50}\n")

    # Initialize database
    init_database()

    # Clear existing data
    with get_connection() as conn:
        conn.execute("DELETE FROM eml_antibiotics")
        conn.execute("DELETE FROM atlas_susceptibility")
        conn.execute("DELETE FROM mic_breakpoints")
        conn.execute("DELETE FROM drug_interactions")
        conn.commit()
    print("Cleared existing data\n")

    # Import all data
    results = {
        "eml_antibiotics": import_eml_antibiotics(),
        "atlas_susceptibility": import_atlas_susceptibility(),
        "mic_breakpoints": import_mic_breakpoints(),
        "drug_interactions": import_drug_interactions(limit=interactions_limit),
    }

    print(f"\n{'='*50}")
    print("Import Summary:")
    for table, count in results.items():
        print(f"  {table}: {count} records")
    print(f"{'='*50}\n")

    return results


if __name__ == "__main__":
    # Import with a limit on interactions for faster demo
    import_all_data(interactions_limit=50000)
