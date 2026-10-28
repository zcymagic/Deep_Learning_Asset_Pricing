/*  Export CRSP monthly stock data for the glass-box SDF panel (OSAP characteristics + CRSP returns).

    Run on WRDS (SAS Studio or a batch job). Output: ~/crsp_msf_1961_2024.csv.gz, about 3.3 million rows.
    Starts in 1961 so that 60-month past-return characteristics are complete from 1967 (CPZ sample start).

    Universe: common stocks (share codes 10, 11) on NYSE, AMEX and NASDAQ (exchange codes 1, 2, 3),
    with share and exchange codes as of each month (from msenames, not the header codes).
    Delisting returns from msedelist are kept in separate columns, so the Python side can
    combine them with RET and apply its own rule for missing delisting returns.

    Columns: permno, yyyymm, date, ret, retx, dlret, dlstcd, prc, shrout, vol, cfacshr, shrcd, exchcd, siccd
    (vol for turnover / volume characteristics, cfacshr for split-adjusted share issuance)
    This is licensed CRSP data: share the output only as your WRDS terms allow.
*/

%let start = '01JAN1961'd;
%let end   = '31DEC2024'd;

/* monthly returns, prices, shares */
proc sql;
    create table msf as
    select a.permno, a.date, a.ret, a.retx, a.prc, a.shrout, a.vol, a.cfacshr
    from crsp.msf as a
    where a.date between &start and &end;
quit;

/* share code, exchange code and SIC code valid on each date */
proc sql;
    create table msf_names as
    select a.*, b.shrcd, b.exchcd, b.siccd
    from msf as a
    inner join crsp.msenames as b
        on a.permno = b.permno
       and b.namedt <= a.date <= b.nameendt
    where b.shrcd in (10, 11)
      and b.exchcd in (1, 2, 3);
quit;

/* delisting returns, matched to the month of delisting */
proc sql;
    create table msf_dl as
    select a.*, d.dlret, d.dlstcd
    from msf_names as a
    left join crsp.msedelist as d
        on a.permno = d.permno
       and intnx('month', a.date, 0, 'E') = intnx('month', d.dlstdt, 0, 'E');
quit;

data out;
    set msf_dl;
    yyyymm = year(date) * 100 + month(date);
    format date yymmddn8.;
    keep permno yyyymm date ret retx dlret dlstcd prc shrout vol cfacshr shrcd exchcd siccd;
run;

proc sort data=out nodupkey;
    by permno yyyymm;
run;

/* gzip-compressed CSV in the home directory (needs SAS 9.4M5 or later for the GZIP option) */
filename outgz zip "~/crsp_msf_1961_2024.csv.gz" gzip;

proc export data=out outfile=outgz dbms=csv replace;
run;

/* quick checks: rows per year and missing returns */
proc sql;
    select int(yyyymm / 100) as year, count(*) as n_rows, nmiss(ret) as n_missing_ret,
           count(distinct permno) as n_stocks
    from out
    group by calculated year;
quit;
