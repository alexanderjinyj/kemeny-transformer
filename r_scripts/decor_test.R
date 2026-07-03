# Run the DECOR consensus-ranking heuristic (ConsRank package) on one test
# dataset and save the resulting rankings incrementally to a wide-format CSV.
#
# Usage:
#   Rscript r_scripts/decor_test.R <dataset_type> [n_samples] [n_voters] [n_items]
#   Rscript r_scripts/decor_test.R random 2000 8 100
#
# Expects data/test/<dataset_type>.csv (samples x (voters*items), no header)
# and writes results/decor/<dataset_type>_decor_rankings.csv.

rm(list = ls(all.names = TRUE))

if (!require(ConsRank)) {
  install.packages("ConsRank")
  library(ConsRank)
}
set.seed(1234)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript r_scripts/decor_test.R <dataset_type> [n_samples] [n_voters] [n_items]")
}
dataset_type <- args[1]
n_samples <- ifelse(length(args) >= 2, as.integer(args[2]), 2000L)
n_voters <- ifelse(length(args) >= 3, as.integer(args[3]), 8L)
n_items <- ifelse(length(args) >= 4, as.integer(args[4]), 100L)

input_file <- file.path("data", "test", paste0("test_dataset_", dataset_type, ".csv"))
output_dir <- file.path("results", "decor")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_filename <- file.path(output_dir, paste0(dataset_type, "_decor_rankings.csv"))

# --- Load and reshape: (samples x voters*items) -> (samples, voters, items) ---
raw <- as.matrix(read.csv(input_file, header = FALSE))
cat("Original matrix dimensions:", dim(raw), "\n")

# R fills arrays column-by-column, so transpose first, then permute dims.
temp_array <- array(t(raw), dim = c(n_items, n_voters, n_samples))
dataset_array <- aperm(temp_array, c(3, 2, 1)) + 1  # DECOR expects 1-indexed ranks
print(dataset_array[1, , ])

cat(paste("\n===== STARTING DECOR FOR:", dataset_type, "=====\n"))

for (i in 1:dim(dataset_array)[1]) {
  res_decor <- DECOR(dataset_array[i, , ], Wk = NULL, NP = 15, L = 100,
                     FF = 0.4, CR = 0.9, FULL = FALSE)

  integer_ranking <- as.integer(round(res_decor$Consensus, 0))
  single_row_df <- as.data.frame(t(integer_ranking))
  names(single_row_df) <- paste0("rank_", 1:length(integer_ranking))
  single_row_df <- cbind(
    data.frame(iteration_id = i, elapsed_time = res_decor$Eltime),
    single_row_df
  )

  # Incremental save: header on first write, append afterwards
  write.table(single_row_df, file = output_filename, sep = ",",
              row.names = FALSE, col.names = !file.exists(output_filename),
              append = file.exists(output_filename))

  if (i %% 500 == 1) {
    cat(paste("... Completed iteration", i, "for", dataset_type,
              "time", res_decor$Eltime, "\n"))
  }
}

cat(paste("===== FINISHED:", dataset_type, "=====\n"))
cat(paste("Results saved to:", output_filename, "\n"))
