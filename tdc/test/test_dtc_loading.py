import pytest

from tdc.multi_pred import DTI

def test_dtc_downloading_and_loading():
    data = DTI(name='dtc_kd')
    assert len(data.get_data()) == 28018
    data = DTI(name = 'dtc_ki')
    assert len(data.get_data()) == 276226
    data = DTI(name = 'dtc_ic50')
    assert len(data.get_data()) == 475362
    data = DTI(name = 'dtc_ec50')
    assert len(data.get_data()) == 63779
