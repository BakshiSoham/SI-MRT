## ============================
## PoSI (fixed-X) on selected set 
## ============================

## ---paths---
repo_dir <- path.expand("~/Downloads/R-master")               # folder containing utilities.R and tmax_1.0.tar.gz
data_dir <- path.expand("~/Documents/git/SI-MRT/selectinf/tests")  # folder with X.csv, y.csv, etc.

## --- sanity checks on repo files ---
util_file <- file.path(repo_dir, "utilities.R")
tmax_tar  <- file.path(repo_dir, "tmax_1.0.tar.gz")
if (!file.exists(util_file)) stop("Cannot find utilities.R at: ", util_file,
                                  "\nFix repo_dir above.")
if (!file.exists(tmax_tar))  warning("tmax_1.0.tar.gz not found at: ", tmax_tar,
                                     "\nIf fixedx_posi() fails, install tmax manually or check repo.")

## --- load/ensure tmax (needed by utilities.R) ---
if (!requireNamespace("tmax", quietly = TRUE)) {
  if (file.exists(tmax_tar)) {
    install.packages(tmax_tar, repos = NULL, type = "source")
  } else {
    stop("Package 'tmax' is missing and bundled tarball not found.\n",
         "Install 'tmax' or place tmax_1.0.tar.gz in repo_dir.")
  }
}
library(tmax)

## --- source utilities.R (brings in fixedx_posi(), posi(), Generate(), etc.) ---
source(util_file)

## --- read data exported from Python ---
X  <- as.matrix(read.csv(file.path(data_dir, "X.csv"), header = FALSE))
y  <- as.numeric(read.csv(file.path(data_dir, "y.csv"), header = FALSE)[, 1])
beta_true <- as.numeric(read.csv(file.path(data_dir, "beta_true.csv"), header = FALSE)[, 1])

## selected_vars can be either (i) 0/1 indicator of length p, or (ii) 1-based indices
sel_raw <- as.numeric(read.csv(file.path(data_dir, "selected_vars.csv"), header = FALSE)[, 1])

## Python also exported the projection targets for the selected set (aligned to sel order)
beta_tgt_sel <- as.numeric(read.csv(file.path(data_dir, "beta_target_selected.csv"), header = FALSE)[, 1])

## --- interpret selection ---
p <- ncol(X)
if (length(sel_raw) == p && all(sel_raw %in% c(0, 1))) {
  sel_idx <- which(sel_raw == 1)           # indicator -> positions
} else {
  sel_idx <- as.integer(sel_raw)           # assume these are 1-based indices
}

if (length(sel_idx) == 0) {
  stop("Selected set is empty (no variables selected).")
}

## --- run fixed-X PoSI on the selected set only ---
conf_level <- 0.95
nboot <- 2000   # increase for final results

fit <- fixedx_posi(X, y, alpha = 1 - conf_level, Nboot = nboot)
ret <- posi(fit, sel_idx)

## posi() sometimes returns a list with $ci, or a matrix directly
ci <- if (!is.null(ret$ci)) ret$ci else as.matrix(ret)
if (!is.matrix(ci) || ncol(ci) != 2 || nrow(ci) != length(sel_idx)) {
  stop("Unexpected CI shape from posi(); check 'str(ret)'.")
}

## --- lengths and coverage vs your Python projection targets ---
ci_lower <- ci[, 1]
ci_upper <- ci[, 2]
len_posi <- ci_upper - ci_lower

## Your main apples-to-apples coverage metric:
## compare to Python’s projection targets (already aligned to sel order)
if (length(beta_tgt_sel) != length(sel_idx)) {
  stop("beta_target_selected length (", length(beta_tgt_sel),
       ") does not match |selected set| (", length(sel_idx), ").")
}
cov_posi_tgt <- (ci_lower <= beta_tgt_sel) & (beta_tgt_sel <= ci_upper)

## --- OPTIONAL: coverage vs projection of true beta onto selected columns ---
## This matches the same projection definition used in Python:
## beta_S^* = (X_S^T X_S)^{-1} X_S^T (X beta_true)
XS <- X[, sel_idx, drop = FALSE]
XtX <- crossprod(XS)
Xtb <- crossprod(XS, X %*% beta_true)
## use a stable solve; add a tiny ridge if XtX is near-singular
beta_proj_true <- tryCatch(
  as.numeric(solve(XtX, Xtb)),
  error = function(e) as.numeric(solve(XtX + diag(1e-8, ncol(XtX)), Xtb))
)
cov_posi_trueproj <- (ci_lower <= beta_proj_true) & (beta_proj_true <= ci_upper)

## --- summarize & write results ---
results <- data.frame(
  var_1based = sel_idx,
  lower      = ci_lower,
  upper      = ci_upper,
  length     = len_posi,
  covered_vs_beta_target = as.integer(cov_posi_tgt),
  covered_vs_true_projection = as.integer(cov_posi_trueproj)
)

cat("\n--- PoSI (fixed-X) summary on selected set ---\n")
cat("Selected variables:", paste(sel_idx, collapse = ", "), "\n")
cat(sprintf("Mean length: %.4f (median: %.4f)\n", mean(len_posi), median(len_posi)))
cat(sprintf("Coverage vs Python projection targets: %.3f\n", mean(cov_posi_tgt)))
cat(sprintf("Coverage vs true-beta projection:     %.3f\n", mean(cov_posi_trueproj)))

out_csv <- file.path(data_dir, "posi_selected_results.csv")
write.csv(results, out_csv, row.names = FALSE)
cat("Wrote:", out_csv, "\n")





# PoSI for selected variables
posi_for_selection <- function(xx, yy, selected_vars, conf_level = 0.95, nboot = 2000) {
  if (is.logical(selected_vars)) selected_vars <- which(selected_vars)
  fit <- fixedx_posi(xx, yy, alpha = 1 - conf_level, Nboot = nboot)
  ret <- posi(fit, selected_vars)
  ci <- if (is.list(ret) && !is.null(ret$ci)) ret$ci else as.matrix(ret)
  data.frame(
    var    = selected_vars,
    lower  = ci[, 1],
    upper  = ci[, 2],
    length = ci[, 2] - ci[, 1]
  )
}

posi_tab <- posi_for_selection(X, y, selected_vars, conf_level = 0.95, nboot = 2000)
print(posi_tab)


# Add coverage column
posi_tab$covered <- (beta_true[posi_tab$var] >= posi_tab$lower) &
  (beta_true[posi_tab$var] <= posi_tab$upper)

coverage <- mean(posi_tab$covered)
cat("Coverage:", coverage, "\n")



## ===== PoSI via CRAN (no compilation), on selected set =====
## Data exported from Python:
##   X.csv, y.csv, beta_true.csv, selected_vars.csv (0/1 indicator or 1-based indices),
##   beta_target_selected.csv (projection targets aligned to selected order)

library(PoSI)

data_dir <- path.expand("~/Documents/git/SI-MRT/selectinf/tests")  # <= set this

X  <- as.matrix(read.csv(file.path(data_dir, "X.csv"), header = FALSE))
y  <- as.numeric(read.csv(file.path(data_dir, "y.csv"), header = FALSE)[,1])
beta_true <- as.numeric(read.csv(file.path(data_dir, "beta_true.csv"), header = FALSE)[,1])
sel_raw <- as.numeric(read.csv(file.path(data_dir, "selected_vars.csv"), header = FALSE)[,1])
beta_tgt_sel <- as.numeric(read.csv(file.path(data_dir, "beta_target_selected.csv"), header = FALSE)[,1])

p <- ncol(X)
if (length(sel_raw) == p && all(sel_raw %in% c(0,1))) {
  sel_idx <- which(sel_raw == 1)                 # indicator -> positions
} else {
  sel_idx <- as.integer(sel_raw)                 # assume 1-based indices
}
stopifnot(length(sel_idx) > 0)

## PoSI object (full design), then subset CIs to selected set
posi_obj <- PoSI(X)                              # CRAN PoSI: no intercept arg
ci_all   <- as.matrix(confint(posi_obj, level = 0.95))
ci       <- ci_all[sel_idx, , drop = FALSE]
stopifnot(ncol(ci) == 2, nrow(ci) == length(sel_idx))

## Lengths and coverage vs the SAME projection targets you used in Python
ci_lower <- ci[,1]; ci_upper <- ci[,2]
len_posi <- ci_upper - ci_lower
stopifnot(length(beta_tgt_sel) == length(sel_idx))
cov_posi <- (ci_lower <= beta_tgt_sel) & (beta_tgt_sel <= ci_upper)

## (Optional) Coverage vs projection of true beta onto selected columns
XS <- X[, sel_idx, drop = FALSE]
beta_proj_true <- tryCatch(
  as.numeric(solve(crossprod(XS), crossprod(XS, X %*% beta_true))),
  error = function(e) as.numeric(solve(crossprod(XS) + diag(1e-8, ncol(XS)), crossprod(XS, X %*% beta_true)))
)
cov_posi_trueproj <- (ci_lower <= beta_proj_true) & (beta_proj_true <= ci_upper)

## Print + save
cat("\n--- CRAN PoSI summary on selected set ---\n")
cat("Selected (1-based):", paste(sel_idx, collapse = ", "), "\n")
cat(sprintf("Mean length: %.4f | Median length: %.4f\n", mean(len_posi), median(len_posi)))
cat(sprintf("Coverage vs Python projection targets: %.3f\n", mean(cov_posi)))
cat(sprintf("Coverage vs true-beta projection:     %.3f\n", mean(cov_posi_trueproj)))

out <- data.frame(
  var_1based = sel_idx,
  lower = ci_lower, upper = ci_upper, length = len_posi,
  covered_vs_beta_target = as.integer(cov_posi),
  covered_vs_true_projection = as.integer(cov_posi_trueproj)
)
write.csv(out, file.path(data_dir, "posi_selected_results_cran.csv"), row.names = FALSE)
cat("Wrote:", file.path(data_dir, "posi_selected_results_cran.csv"), "\n")


