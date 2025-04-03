import pandas as pd
from datetime import datetime

# Rename the columns
df = df.rename(columns={
    'Call': 'Idcall',
    'Work Programme': 'cluster'
})

# Add the new column with the same datetime value for all rows
df['YearSpan'] = datetime(2025, 1, 1)
