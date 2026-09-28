from modelic.balance_sheets import BalanceSheetFactory, IFRS17BalanceSheet, SIIBalanceSheet


def test_sii_balance_sheet_builds():
    sheet = BalanceSheetFactory.create(
        "SII",
        asset_value=200.0,
        liability_value=150.0,
        matching_adjustment=10.0,
        risk_margin=5.0,
        bscr=30.0,
    )
    result = sheet.build()

    assert result.regime == "SII"
    assert result.assets == 200.0
    assert result.liabilities == 150.0
    assert result.surplus == 50.0
    assert result.details["matching_adjustment"] == 10.0


def test_ifrs17_balance_sheet_builds():
    sheet = IFRS17BalanceSheet(
        asset_value=220.0,
        liability_value=160.0,
        csm=20.0,
        risk_adjustment=10.0,
        fulfilment_cashflows=130.0,
    )
    result = sheet.build()

    assert result.regime == "IFRS17"
    assert result.assets == 220.0
    assert result.liabilities == 160.0
    assert result.details["equity"] == 60.0


def test_factory_dispatches_regimes():
    sii = BalanceSheetFactory.create("SII", asset_value=1.0, liability_value=0.5)
    ifrs = BalanceSheetFactory.create("IFRS17", asset_value=1.0, liability_value=0.5)

    assert isinstance(sii, SIIBalanceSheet)
    assert isinstance(ifrs, IFRS17BalanceSheet)

