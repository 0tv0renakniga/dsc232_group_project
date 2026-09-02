Output directories:
    Plots:   /expanse/lustre/projects/uci157/rrogers/data/eda_plots
    Samples: /expanse/lustre/projects/uci157/rrogers/data/sample_data
========================================================================
  HPC SparkSession Configuration
========================================================================
  TOTAL_CORES ............. 32
  TOTAL_MEMORY_GB ......... 128
  DRIVER_MEMORY ........... 10g
  EXECUTOR_CORES .......... 5
  EXECUTOR_INSTANCES ...... 6
  EXECUTOR_MEMORY ......... 19g
  SHUFFLE_PARTITIONS ...... 64
========================================================================

  Discovered 5 dataset(s):
    1. bedmap3_static.parquet
    2. grace.parquet
    3. icesat2_dynamic.parquet
    4. ocean_dynamic.parquet
    5. antarctica_sparse_features.parquet

════════════════════════════════════════════════════════════════════════
  PHASE 1 : Per-Dataset EDA Pipeline
════════════════════════════════════════════════════════════════════════

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DATASET:  bedmap3_static.parquet
  PATH:     /expanse/lustre/projects/uci157/rrogers/data/indiv_data/bedmap3_static.parquet
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [Phase 1] Ingesting + reading schema ...
  Schema (13 columns):
  Column Name                     Data Type           
  ──────────────────────────────  ────────────────────
  x                               double              
  y                               double              
  spatial_ref                     bigint              
  surface                         float               
  bed                             float               
  thickness                       float               
  mask                            tinyint             
  mascon_id                       double              
  bed_slope                       float               
  dist_to_grounding_line          float               
  clamped_depth                   float               
  dist_to_ocean                   float               
  ice_draft                       float               
  [1.5s]

  [Phase 2] Counting rows ...
  Total rows:        54,236,727
  Total columns:             13
  [1.2s]

  [Phase 3] Computing summary statistics ...
  Numeric columns (13): ['x', 'y', 'spatial_ref', 'surface', 'bed', 'thickness', 'mask', 'mascon_id', 'bed_slope', 'dist_to_grounding_line', 'clamped_depth', 'dist_to_ocean', 'ice_draft']

  Summary Statistics for: bedmap3_static.parquet
-RECORD 0-------------------------------------
 summary                | count               
 x                      | 54236727            
 y                      | 54236727            
 spatial_ref            | 54236727            
 surface                | 54188350            
 bed                    | 54236727            
 thickness              | 54236727            
 mask                   | 54236727            
 mascon_id              | 54236727            
 bed_slope              | 54236727            
 dist_to_grounding_line | 54236727            
 clamped_depth          | 6128170             
 dist_to_ocean          | 6153122             
 ice_draft              | 6128170             
-RECORD 1-------------------------------------
 summary                | min                 
 x                      | -2658250.0          
 y                      | -2493750.0          
 spatial_ref            | 0                   
 surface                | 0.9999926           
 bed                    | -3003.0             
 thickness              | 0.0                 
 mask                   | 1                   
 mascon_id              | 201.0               
 bed_slope              | 0.0                 
 dist_to_grounding_line | 0.0                 
 clamped_depth          | 0.0                 
 dist_to_ocean          | 5.9053884           
 ice_draft              | -2483.0             
-RECORD 2-------------------------------------
 summary                | max                 
 x                      | 2750250.0           
 y                      | 2321250.0           
 spatial_ref            | 0                   
 surface                | 4734.0              
 bed                    | 4571.0              
 thickness              | 4757.0              
 mask                   | 3                   
 mascon_id              | 170457.0            
 bed_slope              | 2.0755377           
 dist_to_grounding_line | 307358.1            
 clamped_depth          | 902.3393            
 dist_to_ocean          | 943906.25           
 ice_draft              | 150.0               
-RECORD 3-------------------------------------
 summary                | mean                
 x                      | 457964.8308009073   
 y                      | 82705.50544559961   
 spatial_ref            | 0.0                 
 surface                | 1968.1946818273107  
 bed                    | -19.090939834897412 
 thickness              | 1954.0557107363716  
 mask                   | 1.2268987212299887  
 mascon_id              | 78412.23599349939   
 bed_slope              | 0.04428633084574028 
 dist_to_grounding_line | 6026.085969776766   
 clamped_depth          | 218.05310435404678  
 dist_to_ocean          | 249389.3763039005   
 ice_draft              | -413.99859109652107 
-RECORD 4-------------------------------------
 summary                | stddev              
 x                      | 1154291.1263901023  
 y                      | 1010598.3603795386  
 spatial_ref            | 0.0                 
 surface                | 1173.2229926176399  
 bed                    | 701.4950507231852   
 thickness              | 1055.855700409052   
 mask                   | 0.6342826027741315  
 mascon_id              | 31465.299134711986  
 bed_slope              | 0.06088434571312098 
 dist_to_grounding_line | 27611.899182768768  
 clamped_depth          | 167.7036173823966   
 dist_to_ocean          | 216471.43298194322  
 ice_draft              | 238.63534189047147  

  [7.5s]

  [Phase 4] Collecting metadata ...
      [Step 1/2] Global row count ...
      → 54,236,727 rows
      [Step 2/2] Column completeness ...
  [1.2s]

  [Phase 5] Building histograms ...
      Computing histogram: surface ...
      Computing histogram: bed ...
      Computing histogram: thickness ...
      Computing histogram: bed_slope ...
      Computing histogram: dist_to_grounding_line ...
      Computing histogram: clamped_depth ...
      Computing histogram: ice_draft ...
    → Saved: data/eda_plots/fig_03_histograms_bedmap3_static.png
  [71.6s]

  [Phase 6] Building correlation heatmap ...
      Computing 7x7 correlation matrix ...
    → Saved: data/eda_plots/fig_04_correlation_bedmap3_static.png
  [4.1s]

  [Phase 7] Computing range stats ...
  [1.0s]

  [Phase 8] Generating representative sample ...
      Sampling fraction: 0.005531 (target ~250,000 rows from 54,236,727)
      → Sample written: data/sample_data/bedmap3_static_sample.parquet
  [3.5s]

  ✓ bedmap3_static.parquet complete : 91.63s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DATASET:  grace.parquet
  PATH:     /expanse/lustre/projects/uci157/rrogers/data/indiv_data/grace.parquet
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [Phase 1] Ingesting + reading schema ...
  Schema (9 columns):
  Column Name                     Data Type           
  ──────────────────────────────  ────────────────────
  y                               double              
  x                               double              
  time                            timestamp_ntz       
  WGS84                           bigint              
  land_mask                       float               
  lwe_length                      float               
  mascon_id                       double              
  ocean_mask                      float               
  spatial_ref                     bigint              
  [0.1s]

  [Phase 2] Counting rows ...
  Total rows:        19,839,750
  Total columns:              9
  [0.1s]

  [Phase 3] Computing summary statistics ...
  Numeric columns (8): ['y', 'x', 'WGS84', 'land_mask', 'lwe_length', 'mascon_id', 'ocean_mask', 'spatial_ref']

  Summary Statistics for: grace.parquet
-RECORD 0-------------------------
 summary     | count              
 y           | 19839750           
 x           | 19839750           
 WGS84       | 19839750           
 land_mask   | 4065000            
 lwe_length  | 19839750           
 mascon_id   | 19839750           
 ocean_mask  | 15774750           
 spatial_ref | 19839750           
-RECORD 1-------------------------
 summary     | min                
 y           | -4508087.29346864  
 x           | -4510662.70653136  
 WGS84       | 0                  
 land_mask   | 1.0                
 lwe_length  | -1473.2893         
 mascon_id   | 201.0              
 ocean_mask  | 1.0                
 spatial_ref | 0                  
-RECORD 2-------------------------
 summary     | max                
 y           | 4510662.70653136   
 x           | 4508087.29346864   
 WGS84       | 0                  
 land_mask   | 1.0                
 lwe_length  | 318.55743          
 mascon_id   | 230399.0           
 ocean_mask  | 1.0                
 spatial_ref | 0                  
-RECORD 3-------------------------
 summary     | mean               
 y           | 6623.249768401322  
 x           | 8191.156925136567  
 WGS84       | 0.0                
 land_mask   | 1.0                
 lwe_length  | 0.348350804220669  
 mascon_id   | 158163.0278355322  
 ocean_mask  | 1.0                
 spatial_ref | 0.0                
-RECORD 4-------------------------
 summary     | stddev             
 y           | 2296064.2412417764 
 x           | 2285242.1966471337 
 WGS84       | 0.0                
 land_mask   | 0.0                
 lwe_length  | 28.806726180723775 
 mascon_id   | 53916.543098978196 
 ocean_mask  | 0.0                
 spatial_ref | 0.0                

  [0.8s]

  [Phase 4] Collecting metadata ...
      [Step 1/2] Global row count ...
      → 19,839,750 rows
      [Step 2/2] Column completeness ...
  [2.0s]

  [Phase 5] Building histograms ...
      Computing histogram: lwe_length ...
    → Saved: data/eda_plots/fig_03_histograms_grace.png
  [6.3s]

  [Phase 6] Need >= 2 PHYS_COLUMNS : skipping correlation.

  [Phase 7] Computing range stats ...
  [0.4s]

  [Phase 8] Generating representative sample ...
      Sampling fraction: 0.015121 (target ~250,000 rows from 19,839,750)
      → Sample written: data/sample_data/grace_sample.parquet
  [1.4s]

  ✓ grace.parquet complete : 11.06s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DATASET:  icesat2_dynamic.parquet
  PATH:     /expanse/lustre/projects/uci157/rrogers/data/indiv_data/icesat2_dynamic.parquet
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [Phase 1] Ingesting + reading schema ...
  Schema (8 columns):
  Column Name                     Data Type           
  ──────────────────────────────  ────────────────────
  y                               double              
  x                               double              
  delta_h                         float               
  ice_area                        float               
  h_surface_dynamic               float               
  surface_slope                   float               
  spatial_ref                     bigint              
  time                            timestamp_ntz       
  [0.2s]

  [Phase 2] Counting rows ...
  Total rows:     1,386,866,499
  Total columns:              8
  [0.2s]

  [Phase 3] Computing summary statistics ...
  Numeric columns (7): ['y', 'x', 'delta_h', 'ice_area', 'h_surface_dynamic', 'surface_slope', 'spatial_ref']

  Summary Statistics for: icesat2_dynamic.parquet
-RECORD 0---------------------------------
 summary           | count                
 y                 | 1386866499           
 x                 | 1386866499           
 delta_h           | 1386866499           
 ice_area          | 1386866499           
 h_surface_dynamic | 1386586808           
 surface_slope     | 1382181928           
 spatial_ref       | 1386866499           
-RECORD 1---------------------------------
 summary           | min                  
 y                 | -2142750.0           
 x                 | -2647750.0           
 delta_h           | -63.175518           
 ice_area          | 28734.4              
 h_surface_dynamic | -40.98206            
 surface_slope     | 0.0                  
 spatial_ref       | 0                    
-RECORD 2---------------------------------
 summary           | max                  
 y                 | 2251250.0            
 x                 | 2749750.0            
 delta_h           | 72.98236             
 ice_area          | 1056117.8            
 h_surface_dynamic | 4083.0994            
 surface_slope     | 1.5100615            
 spatial_ref       | 0                    
-RECORD 3---------------------------------
 summary           | mean                 
 y                 | 100702.80517299451   
 x                 | 484456.40172915446   
 delta_h           | 0.04365065201126981  
 ice_area          | 1021863.5000643947   
 h_surface_dynamic | 1980.9560510717363   
 surface_slope     | 0.00742234555780814  
 spatial_ref       | 0.0                  
-RECORD 4---------------------------------
 summary           | stddev               
 y                 | 1010022.8715430193   
 x                 | 1149477.3871542753   
 delta_h           | 0.722332850079902    
 ice_area          | 28568.54396990228    
 h_surface_dynamic | 1175.8764835352185   
 surface_slope     | 0.016757591142793443 
 spatial_ref       | 0.0                  

  [40.0s]

  [Phase 4] Collecting metadata ...
      [Step 1/2] Global row count ...
      → 1,386,866,499 rows
      [Step 2/2] Column completeness ...
  [7.1s]

  [Phase 5] Building histograms ...
      Computing histogram: delta_h ...
      Computing histogram: ice_area ...
      Computing histogram: h_surface_dynamic ...
      Computing histogram: surface_slope ...
    → Saved: data/eda_plots/fig_03_histograms_icesat2_dynamic.png
  [1239.4s]

  [Phase 6] Building correlation heatmap ...
      Computing 4x4 correlation matrix ...
    → Saved: data/eda_plots/fig_04_correlation_icesat2_dynamic.png
  [19.5s]

  [Phase 7] Computing range stats ...
  [6.4s]
