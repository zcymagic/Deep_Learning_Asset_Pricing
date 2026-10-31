"""The 153 characteristics of Jensen, Kelly & Pedersen (2023, "Is There a Replication Crisis in Finance?")
and their 13 themes, from the authors' "Cluster Labels.csv" (github.com/bkelly-lab/ReplicationCrisis,
GlobalFactors). Themes are clusters of characteristics whose factor returns co-move."""

THEMES = {
	'Low Leverage': [
		'age', 'aliq_mat', 'at_be', 'bidaskhl_21d', 'cash_at', 'netdebt_me', 'ni_ivol', 'rd_sale', 'rd5_at',
		'tangibility', 'z_score',
	],
	'Investment': [
		'aliq_at', 'at_gr1', 'be_gr1a', 'capx_gr1', 'capx_gr2', 'capx_gr3', 'coa_gr1a', 'col_gr1a', 'emp_gr1',
		'inv_gr1', 'inv_gr1a', 'lnoa_gr1a', 'mispricing_mgmt', 'ncoa_gr1a', 'nncoa_gr1a', 'noa_gr1a',
		'ppeinv_gr1a', 'ret_60_12', 'sale_gr1', 'sale_gr3', 'saleq_gr1', 'seas_2_5na',
	],
	'Size': [
		'ami_126d', 'dolvol_126d', 'market_equity', 'prc', 'rd_me',
	],
	'Value': [
		'at_me', 'be_me', 'bev_mev', 'chcsho_12m', 'debt_me', 'div12m_me', 'ebitda_mev', 'eq_dur', 'eqnetis_at',
		'eqnpo_12m', 'eqnpo_me', 'eqpo_me', 'fcf_me', 'ival_me', 'netis_at', 'ni_me', 'ocf_me', 'sale_me',
	],
	'Quality': [
		'at_turnover', 'cop_at', 'cop_atl1', 'dgp_dsale', 'gp_at', 'gp_atl1', 'mispricing_perf', 'ni_inc8q',
		'niq_at', 'op_at', 'op_atl1', 'opex_at', 'qmj', 'qmj_growth', 'qmj_prof', 'qmj_safety', 'sale_bev',
	],
	'Low Risk': [
		'beta_60m', 'beta_dimson_21d', 'betabab_1260d', 'betadown_252d', 'earnings_variability', 'ivol_capm_21d',
		'ivol_capm_252d', 'ivol_ff3_21d', 'ivol_hxz4_21d', 'ocfq_saleq_std', 'rmax1_21d', 'rmax5_21d', 'rvol_21d',
		'seas_6_10na', 'turnover_126d', 'zero_trades_126d', 'zero_trades_21d', 'zero_trades_252d',
	],
	'Debt Issuance': [
		'capex_abn', 'debt_gr3', 'fnl_gr1a', 'ncol_gr1a', 'nfna_gr1a', 'ni_ar1', 'noa_at',
	],
	'Seasonality': [
		'corr_1260d', 'coskew_21d', 'dbnetis_at', 'kz_index', 'lti_gr1a', 'pi_nix', 'seas_11_15an',
		'seas_11_15na', 'seas_16_20an', 'seas_2_5an', 'seas_6_10an', 'sti_gr1a',
	],
	'Accruals': [
		'cowc_gr1a', 'oaccruals_at', 'oaccruals_ni', 'seas_16_20na', 'taccruals_at', 'taccruals_ni',
	],
	'Profitability': [
		'dolvol_var_126d', 'ebit_bev', 'ebit_sale', 'f_score', 'ni_be', 'niq_be', 'o_score', 'ocf_at', 'ope_be',
		'ope_bel1', 'turnover_var_126d',
	],
	'Profit Growth': [
		'dsale_dinv', 'dsale_drec', 'dsale_dsga', 'niq_at_chg1', 'niq_be_chg1', 'niq_su', 'ocf_at_chg1',
		'ret_12_7', 'sale_emp_gr1', 'saleq_su', 'seas_1_1an', 'tax_gr1a',
	],
	'Short-Term Reversal': [
		'iskew_capm_21d', 'iskew_ff3_21d', 'iskew_hxz4_21d', 'ret_1_0', 'rmax5_rvol_21d', 'rskew_21d',
	],
	'Momentum': [
		'prc_highprc_252d', 'resff3_12_1', 'resff3_6_1', 'ret_12_1', 'ret_3_1', 'ret_6_1', 'ret_9_1', 'seas_1_1na',
	],
}

JKP153 = [c for chars in THEMES.values() for c in chars]

# Direction of each characteristic's long-short factor (+1: high minus low), from the authors'
# "Factor Details.xlsx": the sign with which the characteristic predicts returns in the original study.
DIRECTION = {
	'age': -1, 'aliq_mat': -1, 'at_be': -1, 'bidaskhl_21d': 1, 'cash_at': 1, 'netdebt_me': -1, 'ni_ivol': 1,
	'rd_sale': 1, 'rd5_at': 1, 'tangibility': 1, 'z_score': 1, 'aliq_at': -1, 'at_gr1': -1, 'be_gr1a': -1,
	'capx_gr1': -1, 'capx_gr2': -1, 'capx_gr3': -1, 'coa_gr1a': -1, 'col_gr1a': -1, 'emp_gr1': -1, 'inv_gr1':
	-1, 'inv_gr1a': -1, 'lnoa_gr1a': -1, 'mispricing_mgmt': 1, 'ncoa_gr1a': -1, 'nncoa_gr1a': -1, 'noa_gr1a':
	-1, 'ppeinv_gr1a': -1, 'ret_60_12': -1, 'sale_gr1': -1, 'sale_gr3': -1, 'saleq_gr1': -1, 'seas_2_5na': -1,
	'ami_126d': 1, 'dolvol_126d': -1, 'market_equity': -1, 'prc': -1, 'rd_me': 1, 'at_me': 1, 'be_me': 1,
	'bev_mev': 1, 'chcsho_12m': -1, 'debt_me': 1, 'div12m_me': 1, 'ebitda_mev': 1, 'eq_dur': -1, 'eqnetis_at':
	-1, 'eqnpo_12m': 1, 'eqnpo_me': 1, 'eqpo_me': 1, 'fcf_me': 1, 'ival_me': 1, 'netis_at': -1, 'ni_me': 1,
	'ocf_me': 1, 'sale_me': 1, 'at_turnover': 1, 'cop_at': 1, 'cop_atl1': 1, 'dgp_dsale': 1, 'gp_at': 1,
	'gp_atl1': 1, 'mispricing_perf': 1, 'ni_inc8q': 1, 'niq_at': 1, 'op_at': 1, 'op_atl1': 1, 'opex_at': 1,
	'qmj': 1, 'qmj_growth': 1, 'qmj_prof': 1, 'qmj_safety': 1, 'sale_bev': 1, 'beta_60m': -1,
	'beta_dimson_21d': -1, 'betabab_1260d': -1, 'betadown_252d': -1, 'earnings_variability': -1,
	'ivol_capm_21d': -1, 'ivol_capm_252d': -1, 'ivol_ff3_21d': -1, 'ivol_hxz4_21d': -1, 'ocfq_saleq_std': -1,
	'rmax1_21d': -1, 'rmax5_21d': -1, 'rvol_21d': -1, 'seas_6_10na': -1, 'turnover_126d': -1,
	'zero_trades_126d': 1, 'zero_trades_21d': 1, 'zero_trades_252d': 1, 'capex_abn': -1, 'debt_gr3': -1,
	'fnl_gr1a': -1, 'ncol_gr1a': -1, 'nfna_gr1a': 1, 'ni_ar1': 1, 'noa_at': -1, 'corr_1260d': -1, 'coskew_21d':
	-1, 'dbnetis_at': -1, 'kz_index': 1, 'lti_gr1a': -1, 'pi_nix': 1, 'seas_11_15an': 1, 'seas_11_15na': -1,
	'seas_16_20an': 1, 'seas_2_5an': 1, 'seas_6_10an': 1, 'sti_gr1a': 1, 'cowc_gr1a': -1, 'oaccruals_at': -1,
	'oaccruals_ni': -1, 'seas_16_20na': -1, 'taccruals_at': -1, 'taccruals_ni': -1, 'dolvol_var_126d': -1,
	'ebit_bev': 1, 'ebit_sale': 1, 'f_score': 1, 'ni_be': 1, 'niq_be': 1, 'o_score': -1, 'ocf_at': 1, 'ope_be':
	1, 'ope_bel1': 1, 'turnover_var_126d': -1, 'dsale_dinv': 1, 'dsale_drec': -1, 'dsale_dsga': 1,
	'niq_at_chg1': 1, 'niq_be_chg1': 1, 'niq_su': 1, 'ocf_at_chg1': 1, 'ret_12_7': 1, 'sale_emp_gr1': 1,
	'saleq_su': 1, 'seas_1_1an': 1, 'tax_gr1a': 1, 'iskew_capm_21d': -1, 'iskew_ff3_21d': -1, 'iskew_hxz4_21d':
	-1, 'ret_1_0': -1, 'rmax5_rvol_21d': -1, 'rskew_21d': -1, 'prc_highprc_252d': 1, 'resff3_12_1': 1,
	'resff3_6_1': 1, 'ret_12_1': 1, 'ret_3_1': 1, 'ret_6_1': 1, 'ret_9_1': 1, 'seas_1_1na': 1,
}
THEME_OF = {c: theme for theme, chars in THEMES.items() for c in chars}

# JKP's Value theme without its net-issuance measures (chcsho_12m, eqnetis_at, eqnpo_12m, netis_at):
# the price-scaled ratios, used as the "value" group in Paper 2.
VALUE_RATIOS = [c for c in THEMES['Value'] if c not in ('chcsho_12m', 'eqnetis_at', 'eqnpo_12m', 'netis_at')]
