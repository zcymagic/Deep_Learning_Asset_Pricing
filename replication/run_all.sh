#!/usr/bin/env bash
# Replication driver for the glass-box SDF / auditor / Rashomon results (docs/*_results.md, docs/robustness.md).
#
#   replication/run_all.sh                 # every stage, in order
#   replication/run_all.sh cpz osap        # selected stages, in the order given
#
# Stages (see replication/README.md for inputs, outputs and runtimes):
#   tests synthetic cpz cpz_robustness osap_build osap osap_robustness jkp_build jkp consistency compare
#   jkp_noclip is optional and not part of the default list.
#
# Environment variables:
#   SKIP_NEURAL=1     skip the CPZ-style neural rung (about 8.5 CPU hours; only its ladder/spanning rows need it)
#   FF_DOWNLOAD=1     download Fama-French factors afresh instead of using the copies in replication/factors
#                     (the library revises its data every year, so factor-model numbers then drift slightly)
#   JKP_ARCHIVE=path  the JKP archive holding kelly.csv (default data_raw/kelly.tar.bz2)
#   JKP_CACHE=path    optional cache of the raw JKP extract (about 2 GB) so jkp_build can be re-run quickly
#   PYTHON=python3    Python interpreter
#
# Each step's output goes to replication_logs/<step>.log; a failing step stops the run.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=${PYTHON:-python3}
LOGS=replication_logs
CFG=replication_configs		# derived configs written by this script
mkdir -p "$LOGS" "$CFG" results datasets/factors
ALL=(tests synthetic cpz cpz_robustness osap_build osap osap_robustness jkp_build jkp consistency compare)

run() {	# run <step-name> <command...>
	local name=$1; shift
	echo "[$(date '+%F %T')] $name"
	if ! { time "$@"; } > "$LOGS/$name.log" 2>&1; then
		echo "FAILED: $name (see $LOGS/$name.log)"; exit 1
	fi
	grep '^real' "$LOGS/$name.log" | tail -1 | sed 's/^/    /'
}

derive_config() {	# derive_config <src.json> <dst.json> <python statement on c>
	"$PY" - "$1" "$2" "$3" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
exec(sys.argv[3])
json.dump(c, open(sys.argv[2], 'w'), indent='\t')
EOF
}

factors() {	# factors <config> <out csv>
	if [ "${FF_DOWNLOAD:-0}" = "1" ]; then
		run "factors_$(basename "$2" .csv)" "$PY" scripts/download_ff_factors.py --config "$1" --out "$2"
	else
		cp "replication/factors/$(basename "$2")" "$2"
	fi
}

stage_tests() {
	run tests "$PY" -m pytest -q tests
}

stage_synthetic() {
	run make_synthetic "$PY" scripts/make_synthetic.py
	run paper1_synthetic "$PY" scripts/run_paper1.py --config config_glassbox/synthetic.json --out results/paper1_synthetic
	run paper2_synthetic "$PY" scripts/run_paper2.py --config config_glassbox/synthetic.json --out results/paper2_synthetic
}

stage_cpz() {
	factors config_glassbox/cpz.json datasets/factors/ff5_umd.csv
	local neural=(); [ "${SKIP_NEURAL:-0}" = "1" ] && neural=(--skip_neural)
	run paper1_cpz "$PY" scripts/run_paper1.py --config config_glassbox/cpz.json --out results/paper1_cpz_neural "${neural[@]}"
	run spanning_cpz_gan "$PY" scripts/spanning_vs_external.py --config config_glassbox/cpz.json \
		--models results/paper1_cpz_neural --out results/spanning_cpz_gan
	run paper2_cpz_additive "$PY" scripts/run_paper2.py --config config_glassbox/cpz.json \
		--out results/paper2_cpz_additive --rung additive
	# the additive + state Rashomon run used 100 bootstrap refits instead of the config's 500
	derive_config config_glassbox/cpz.json "$CFG/cpz_p2_state.json" "c['rashomon']['n_boot'] = 100"
	run paper2_cpz_additive_state "$PY" scripts/run_paper2.py --config "$CFG/cpz_p2_state.json" \
		--out results/paper2_cpz_additive_state --rung additive_state
	run combined_cpz "$PY" scripts/run_combined.py --config config_glassbox/cpz.json \
		--models results/paper1_cpz_neural --out results/combined_cpz
	run state_robustness_cpz "$PY" scripts/run_state_robustness.py --config config_glassbox/cpz.json \
		--models results/paper1_cpz_neural --out results/state_robustness_cpz
}

stage_cpz_robustness() {
	run knot_threshold_cpz "$PY" scripts/run_knot_threshold_robustness.py --config config_glassbox/cpz.json \
		--models results/paper1_cpz_neural --out results/knot_threshold_cpz
	run window_friction_cpz "$PY" scripts/run_window_friction_robustness.py --config config_glassbox/cpz.json \
		--models results/paper1_cpz_neural --out results/window_friction_cpz
}

stage_osap_build() {
	run build_osap_median "$PY" scripts/build_osap_panel.py --impute median --out datasets/osap
	run build_osap_cc "$PY" scripts/build_osap_panel.py --impute none --out datasets/osap_cc
}

stage_osap() {
	factors config_glassbox/osap.json datasets/factors/ff5_umd_osap.csv
	factors config_glassbox/osap_post.json datasets/factors/ff5_umd_osap_post.csv
	run paper1_osap "$PY" scripts/run_paper1.py --config config_glassbox/osap.json --out results/paper1_osap --skip_neural
	run paper1_osap_post "$PY" scripts/run_paper1.py --config config_glassbox/osap_post.json \
		--out results/paper1_osap_post_only --skip_neural --reuse results/paper1_osap --holdout test
	run paper2_osap_additive "$PY" scripts/run_paper2.py --config config_glassbox/osap.json \
		--out results/paper2_osap_additive --rung additive
	mkdir -p results/rule_alpha
	run rule_alpha "$PY" scripts/plot_rule_alpha.py --out results/rule_alpha
}

stage_osap_robustness() {
	run knot_threshold_osap "$PY" scripts/run_knot_threshold_robustness.py --config config_glassbox/osap.json \
		--models results/paper1_osap --out results/knot_threshold_osap
	run window_friction_osap "$PY" scripts/run_window_friction_robustness.py --config config_glassbox/osap.json \
		--models results/paper1_osap --out results/window_friction_osap
}

jkp_stream() {	# jkp_stream <archive> <cache args...>: one pass over kelly.csv feeds the sample and the panel
	local archive=$1; shift
	local fifo; fifo=$(mktemp -u); mkfifo "$fifo"
	"$PY" scripts/extract_jkp.py --out datasets/jkp/jkp_sample.npz < "$fifo" & local pid=$!
	tar -xjOf "$archive" kelly.csv | tee "$fifo" | "$PY" scripts/build_jkp_panel.py --out datasets/jkp_panel "$@"
	wait "$pid"; local rc=$?; rm -f "$fifo"; return $rc
}

stage_jkp_build() {
	local cache=()
	[ -n "${JKP_CACHE:-}" ] && cache=(--cache "$JKP_CACHE")
	run build_jkp jkp_stream "${JKP_ARCHIVE:-data_raw/kelly.tar.bz2}" "${cache[@]}"
	run build_jkp_themes "$PY" scripts/build_jkp_theme_panel.py --src datasets/jkp_panel --out datasets/jkp_themes
}

stage_jkp() {
	[ -f datasets/factors/ff5_umd_osap.csv ] || factors config_glassbox/osap.json datasets/factors/ff5_umd_osap.csv
	[ -f datasets/factors/ff5_umd_osap_post.csv ] || factors config_glassbox/osap_post.json datasets/factors/ff5_umd_osap_post.csv
	run paper1_jkp "$PY" scripts/run_paper1.py --config config_glassbox/jkp.json --out results/paper1_jkp --skip_neural
	run paper1_jkp_post "$PY" scripts/run_paper1.py --config config_glassbox/jkp_post.json \
		--out results/paper1_jkp_post --skip_neural --reuse results/paper1_jkp --holdout test
	run paper2_jkp_additive "$PY" scripts/run_paper2.py --config config_glassbox/jkp.json \
		--out results/paper2_jkp_additive --rung additive
	run paper1_jkp_themes "$PY" scripts/run_paper1.py --config config_glassbox/jkp_themes.json \
		--out results/paper1_jkp_themes --skip_neural
	run paper1_jkp_themes_post "$PY" scripts/run_paper1.py --config config_glassbox/jkp_themes_post.json \
		--out results/paper1_jkp_themes_post --skip_neural --reuse results/paper1_jkp_themes --holdout test
	run paper2_jkp_themes_additive "$PY" scripts/run_paper2.py --config config_glassbox/jkp_themes.json \
		--out results/paper2_jkp_themes_additive --rung additive
	run cross_audit_themes "$PY" scripts/audit_cross_panel.py --models results/paper1_jkp_themes \
		--model_config config_glassbox/jkp_themes.json --audit_config config_glassbox/jkp.json \
		--out results/cross_audit_themes_on_jkp153
}

stage_jkp_noclip() {	# optional: the 153-characteristic ladder without state clipping (training failure)
	derive_config config_glassbox/jkp.json "$CFG/jkp_noclip.json" "c.pop('state_clip', None)"
	run paper1_jkp_noclip "$PY" scripts/run_paper1.py --config "$CFG/jkp_noclip.json" --out results/paper1_jkp_noclip --skip_neural
}

stage_consistency() {
	run char_consistency "$PY" scripts/compare_cpz_osap_chars.py --out results/char_consistency
	run char_crossvalidation "$PY" scripts/compare_cpz_osap_jkp.py --jkp datasets/jkp/jkp_sample.npz \
		--out results/char_crossvalidation
}

stage_compare() {
	"$PY" replication/collect_and_compare.py --out replication_output
}

stages=("$@"); [ ${#stages[@]} -eq 0 ] && stages=("${ALL[@]}")
for st in "${stages[@]}"; do
	declare -F "stage_$st" > /dev/null || { echo "unknown stage: $st"; exit 2; }
done
"$PY" replication/check_inputs.py "${stages[@]}" || { echo "inputs missing; see replication/README.md"; exit 1; }
for st in "${stages[@]}"; do
	echo "=== stage $st"
	"stage_$st"
done
echo "done; logs in $LOGS/"
