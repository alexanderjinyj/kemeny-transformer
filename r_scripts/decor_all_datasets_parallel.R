rm(list = ls(all.names = TRUE))

# --- 1. SETUP: Load libraries ---
library(ConsRank)
library(parallel)
set.seed(1234)

# --- 2. CONFIGURATION ---
# Define all datasets to process
# Each entry: list(csv_path, num_voters, num_items, num_batches, dataset_type, label)
datasets <- list()

dataset_types <- c("random", "repeat", "jiggling")
num_voters_list <- c(6, 8, 10)
num_items_list <- c(90, 100, 110, 125, 150)
num_batches <- 500

for (dtype in dataset_types) {
  for (nv in num_voters_list) {
    for (ni in num_items_list) {
      csv_path <- paste0("test_dataset/test_dataset_", dtype,
                         "/test_dataset_", dtype, "_nvoters_", nv,
                         "_nitems_", ni, ".csv")
      label <- paste0(dtype, "_v", nv, "_i", ni)
      
      datasets[[label]] <- list(
        csv_path = csv_path,
        num_voters = nv,
        num_items = ni,
        num_batches = num_batches,
        dataset_type = dtype,
        label = label
      )
    }
  }
}

# --- 3. DEFINE THE PROCESSING FUNCTION ---
process_single_dataset <- function(config) {
  csv_path <- config$csv_path
  nv <- config$num_voters
  ni <- config$num_items
  nb <- config$num_batches
  label <- config$label
  dtype <- config$dataset_type
  
  output_dir <- paste0("test_dataset/test_dataset_", dtype, "/decor_result")
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  output_filename <- paste0(output_dir, "/test_dataset_", label, "_decor_rankings.csv")

  # Skip if output already exists
  if (file.exists(output_filename)) {
    cat(paste("[SKIP]", label, "- output already exists:", output_filename, "\n"))
    return(paste("SKIPPED:", label))
  }
  
  # Check if input CSV exists
  if (!file.exists(csv_path)) {
    cat(paste("[ERROR]", label, "- CSV not found:", csv_path, "\n"))
    return(paste("ERROR:", label, "- CSV not found"))
  }
  
  cat(paste("[START]", label, "- Loading", csv_path, "\n"))
  
  # Load and reshape data
  raw_matrix <- as.matrix(read.csv(csv_path, header = FALSE))
  transposed_matrix <- t(raw_matrix)
  temp_array <- array(transposed_matrix, dim = c(ni, nv, nb))
  data_array <- aperm(temp_array, c(3, 2, 1))
  
  # Add 1 for R's 1-based indexing
  data_array <- data_array + 1
  
  cat(paste("[PROCESSING]", label, "- Shape:", dim(data_array)[1], "x",
            dim(data_array)[2], "x", dim(data_array)[3], "\n"))
  
  # Process each batch
  for (i in 1:nb) {
    res_decor <- DECOR(data_array[i, , ], Wk = NULL, NP = 15, L = 100,
                       FF = 0.4, CR = 0.9, FULL = FALSE)
    
    consensus_vector <- res_decor$Consensus
    integer_ranking <- as.integer(round(consensus_vector, 0))
    
    single_row_df <- as.data.frame(t(integer_ranking))
    names(single_row_df) <- paste0("rank_", 1:length(integer_ranking))
    
    single_row_df <- cbind(
      data.frame(iteration_id = i, elapsed_time = res_decor$Eltime),
      single_row_df
    )
    
    if (!file.exists(output_filename)) {
      write.table(single_row_df, file = output_filename, sep = ",",
                  row.names = FALSE, col.names = TRUE)
    } else {
      write.table(single_row_df, file = output_filename, sep = ",",
                  row.names = FALSE, col.names = FALSE, append = TRUE)
    }
    
    if (i %% 100 == 1 || i == nb) {
      cat(paste("  [", label, "] Batch", i, "/", nb,
                "- Time:", round(res_decor$Eltime, 2), "\n"))
    }
  }
  
  cat(paste("[DONE]", label, "-> Saved to:", output_filename, "\n"))
  return(paste("COMPLETED:", label))
}

# --- 4. RUN IN PARALLEL ---
# Detect number of available cores, use at most the number of datasets
num_cores <- min(detectCores() - 1, length(datasets))
cat(paste("\n=== Running DECOR on", length(datasets), "datasets using",
          num_cores, "CPU cores ===\n\n"))

results <- mclapply(datasets, process_single_dataset, mc.cores = num_cores)

# --- 5. SUMMARY ---
cat("\n\n=== SUMMARY ===\n")
for (r in results) {
  cat(paste(" ", r, "\n"))
}
cat("All datasets have been processed.\n")
