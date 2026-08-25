from pathlib import Path
import pytest
from delay_intelligence.data.adapters.olist import OlistAdapter

DATA = Path('data/external/olist/olist_orders_dataset.csv')
pytestmark = pytest.mark.skipif(not DATA.exists(), reason='Olist is not bundled; external validation is NOT VALIDATED')


def test_olist_adapter_loads_and_maps_ontology():
    adapter = OlistAdapter()
    df = adapter.load_and_transform(nrows=100)
    assert not df.empty
    assert 'T_pred' in df.columns
    assert 'Delay_Flag' in df.columns
    assert 'Country' in df.columns
    assert df['Country'].iloc[0] == 'Brazil'


def test_olist_eligibility():
    adapter = OlistAdapter()
    df = adapter.load_and_transform(nrows=100)
    assert 'delivered' in df['order_status'].values
