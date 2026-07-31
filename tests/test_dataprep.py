import pandas as pd

from scripts.data_prep.dataprep import has_explicit_ec_number


def test_has_explicit_ec_number_accepts_supported_biolip_formats():
    values = pd.Series(
        [
            "2.1.1.37",
            "3.1.-.-",
            "4.2.2.n1",
            "?,3.1.3.16",
            "'3.6.1.11,3.6.1.40'",
        ]
    )

    assert has_explicit_ec_number(values).tolist() == [True] * len(values)


def test_has_explicit_ec_number_rejects_unknown_or_malformed_values():
    values = pd.Series(
        [
            "?",
            "",
            None,
            pd.NA,
            "not assigned",
            "GO:0003824",
            "1.2.3",
            "12.3.4.5",
            "A1.2.3.4B",
        ]
    )

    assert has_explicit_ec_number(values).tolist() == [False] * len(values)
