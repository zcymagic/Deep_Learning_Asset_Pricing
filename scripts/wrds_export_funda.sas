/*  Export Compustat annual fundamentals for the CPZ characteristics that OSAP does not reproduce.

    Run on WRDS (SAS Studio or a batch job). Output: ~/comp_funda_1958_2024.csv.gz, about 0.5 million rows.

    Purpose: rebuild, to CPZ's (Freyberger, Neuhierl & Weber 2020) definitions, the characteristics whose
    OSAP versions disagree with CPZ (docs/osap_results.md, "Consistency with the CPZ characteristics"):
        AC, OA, CF, D2A, NI
    and the four with no OSAP counterpart:
        Q, SGA2S, FC2Y, PCM
    The remaining items (book equity, sales, costs, cash flow) let other accounting characteristics be
    checked against OSAP too.

    Starts in fiscal year 1958: changes need the prior year, and accounting data enter the panel with a
    lag, so 1958 covers the CPZ sample start in 1967 with room to spare.

    Universe: standard industrial-format records (indfmt INDL, datafmt STD, popsrc D, consol C), linked to
    CRSP permno through the CCM link history (link types LU and LC, primary links P and C) valid at the
    fiscal year end. Records without a valid link are dropped.

    Columns: gvkey, lpermno (CRSP permno), datadate, fyear, fyr, curcd and the items listed in %items.
    This is licensed Compustat data: share the output only as your WRDS terms allow.
*/

%let start_fy = 1958;
%let end_fy   = 2024;

%let items =
    at act che lct dlc dltt lt txp txdb txditc
    ceq seq pstk pstkrv pstkl
    sale revt cogs xsga xrd dp ppent ppegt invt rect ap
    ib ni oancf wcapch capx
    csho ajex prcc_f dvc dvt;

/* annual fundamentals */
data funda;
    set comp.funda(keep=gvkey datadate fyear fyr indfmt datafmt popsrc consol curcd &items);
    where indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' and consol = 'C'
      and &start_fy <= fyear <= &end_fy;
    drop indfmt datafmt popsrc consol;
run;

/* CRSP-Compustat link valid at the fiscal year end */
proc sql;
    create table funda_ccm as
    select b.lpermno, b.linkprim, a.*
    from funda as a
    inner join crsp.ccmxpf_lnkhist as b
        on a.gvkey = b.gvkey
       and b.linktype in ('LU', 'LC')
       and b.linkprim in ('P', 'C')
       and b.linkdt <= a.datadate
       and (a.datadate <= b.linkenddt or missing(b.linkenddt));
quit;

data out;
    set funda_ccm;
    format datadate yymmddn8.;
run;

/* one record per permno and fiscal year end: when a permno links to two gvkeys, keep the primary link (P) */
proc sort data=out;
    by lpermno datadate descending linkprim;
run;

proc sort data=out nodupkey;
    by lpermno datadate;
run;

data out;
    set out;
    drop linkprim;
run;

/* gzip-compressed CSV in the home directory (needs SAS 9.4M5 or later for the GZIP option) */
filename outgz zip "~/comp_funda_1958_2024.csv.gz" gzip;

proc export data=out outfile=outgz dbms=csv replace;
run;

/* quick checks: firms per fiscal year and coverage of the key items */
proc sql;
    select fyear, count(*) as n_rows, count(distinct lpermno) as n_permnos,
           nmiss(at) as miss_at, nmiss(dp) as miss_dp, nmiss(oancf) as miss_oancf,
           nmiss(xsga) as miss_xsga, nmiss(ajex) as miss_ajex
    from out
    group by fyear;
quit;
