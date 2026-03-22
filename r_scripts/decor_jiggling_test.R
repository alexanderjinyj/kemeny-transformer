rm(list = ls(all.names = TRUE))
#install.packages("ConsRank")
# --- 1. SETUP: Load libraries and set seed ---
if (!require(ConsRank)) {
  install.packages("ConsRank")
}
library(ConsRank)
set.seed(1234)


# --- 2. DATA LOADING: Load and reshape your datasets ---
# Ensure your CSV files are in a sub-folder named "test_dataset"
test_dataset_jiggling <- read.csv("test_dataset/test_dataset_jiggling.csv", header = FALSE)

test_dataset_jiggling_matrix <- as.matrix(test_dataset_jiggling)

cat("Original matrix dimensions:", dim(test_dataset_jiggling_matrix), "\n") # Should be 10000 x 800

# --- EFFICIENT RESHAPING ---

# Step 2: Transpose the entire matrix from (10000 x 800) to (800 x 10000)
transposed_matrix <- t(test_dataset_jiggling_matrix)

# Step 3: Reshape the transposed data into an array.
# R fills this array column-by-column, which now corresponds to our original rows.
# The dimensions are set to (columns_in_submatrix, rows_in_submatrix, number_of_records)
# So, (100, 8, 10000)
temp_array <- array(transposed_matrix, dim = c(100, 8, 10000))

# Step 4: Permute (reorder) the dimensions to the final desired shape.
# We want to move the 3rd dimension (10000) to the front.
# We map (100, 8, 10000) -> (10000, 8, 100)
# So the order becomes (3, 2, 1)
test_dataset_jiggling_array <- aperm(temp_array, c(3, 2, 1))

# --- 3. DATA PREPARATION: Adjust all values in the arrays by adding 1 ---
cat("Adjusting all data points by adding 1...\n")
test_dataset_jiggling_array <- test_dataset_jiggling_array + 1
print(test_dataset_jiggling_array[1, , ]) # Print the first slice to verify

# --- 4. BATCH PROCESSING: Loop through each dataset and save wide-format results ---

# Create a named list of the datasets to loop through them efficiently
datasets_to_process <- list(
  jiggling_data = test_dataset_jiggling_array
)

for (dataset_name in names(datasets_to_process)) {
  
  cat(paste("\n\n===== STARTING PROCESSING FOR:", dataset_name, "=====\n"))
  
  current_array <- datasets_to_process[[dataset_name]]
  output_filename <- paste0(dataset_name, "_decor_test_result_rankings.csv")
  # Create an empty list to store the single-row data frame from each iteration

  # ===== INNER LOOP: Iterates through each of the 10,000 slices =====
  for (i in 1:dim(current_array)[1]) {
    
    # Run the DECOR function
    res_decor <- DECOR(current_array[i, , ], Wk = NULL, NP = 15, L = 100, FF = 0.4, CR = 0.9, FULL = FALSE)
    
    # --- Process the results into the desired format ---
    
    # 1. Get the 100-element consensus ranking vector
    consensus_vector <- res_decor$Consensus
    
    # 2. Round to the nearest whole number and convert to integer type
    integer_ranking <- as.integer(round(consensus_vector, 0))
    
   # Create a one-row data frame
    single_row_df <- as.data.frame(t(integer_ranking))
    names(single_row_df) <- paste0("rank_", 1:length(integer_ranking))
    
    # Add the iteration ID and time to the front
    single_row_df <- cbind(
      data.frame(iteration_id = i, elapsed_time = res_decor$Eltime),
      single_row_df
    )
    
    # --- WRITE TO CSV: The key logic for incremental saving ---
    
     if (!file.exists(output_filename)) {
      # For the FIRST iteration, create the file and write the header
        write.table(
      single_row_df,
      file = output_filename,
      sep = ",",           
      row.names = FALSE,   
      col.names = TRUE,
    )
    } else {
      # For ALL OTHER iterations, APPEND the data without the header
      write.table(
        single_row_df,
        file = output_filename,
        sep = ",",
        row.names = FALSE,
        col.names = FALSE,
        append = TRUE
      )
    } 
    
    # Print progress to the console
    if ((i %% 500 == 1)) {
      cat(paste("... Completed iteration", i, "for", dataset_name,"time", res_decor$Eltime, "\n"))
    }
    
    # Print progress to the console
    if (i %% 500 == 1) {
      cat(paste("... Completed iteration", i, "for", dataset_name,"time", res_decor$Eltime, "\n"))
    }
  } # --- End of inner loop ---
  
  
  cat(paste("===== FINISHED:", dataset_name, "=====\n"))
  cat(paste("All 10,000 results saved to:", output_filename, "\n"))
  
} # --- End of outer loop ---

cat("\n\nAll datasets have been processed and saved.\n")```


# ===== OUTER LOOP: Iterates through each dataset =====

# --- End of outer loop ---

cat("\n\nAll datasets have been processed and saved.\n")```

### What the Output CSV Will Look Like

This script will generate three files:
1.  `jiggling_data_wide_rankings.csv`
2.  `random_data_wide_rankings.csv`
3.  `jiggling_data_wide_rankings.csv`

Each file will have **10,000 rows** (one for each iteration) and **102 columns**:

| iteration\_id | elapsed\_time | rank\_1 | rank\_2 | rank\_3 | ... | rank\_100 |
| :------------ | :------------ | :------ | :------ | :------ | :-- | :-------- |
| 1             | 15.23         | 5       | 2       | 8       | ... | 1         |
| 2             | 14.98         | 3       | 7       | 1       | ... | 4         |
| 3             | 15.41         | 8       | 2       | 5       | ... | 6         |
| ...           | ...           | ...     | ...     | ...     | ... | ...       |
| 10000         | 16.05         | 1       | 6       | 4       | ... | 3         |