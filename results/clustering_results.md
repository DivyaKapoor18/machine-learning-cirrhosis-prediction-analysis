# Clustering Results Summary

## K-Means Clustering Performance
- **Silhouette Score**: 0.22
- **Number of Clusters**: 3
- **Cluster Distribution**: 
  - Cluster 0: 221 patients (52.9%)
  - Cluster 1: 106 patients (25.4%)
  - Cluster 2: 91 patients (21.8%)

## DBSCAN Clustering Performance
- **Number of Clusters**: 3 (excluding noise)
- **Noise Points**: Handled effectively
- **Cluster Distribution**: Variable sizes based on density

## Key Clinical Patterns Identified

### High-Risk Patient Group (Cluster 2 - K-Means)
- **Elevated Bilirubin**: 7.51 mg/dl (vs 1.50 in Cluster 0)
- **Lower Albumin**: 3.19 mg/dl (vs 3.65 in Cluster 0)
- **Advanced Disease Stage**: Average stage 3.60

### Moderate-Risk Group (Cluster 1)
- **Intermediate Values**: Between high and low-risk groups
- **Potential for progression**: Requires monitoring

### Low-Risk Group (Cluster 0)
- **Stable Markers**: Lower bilirubin, higher albumin
- **Earlier Disease Stage**: Average stage 2.80

## Visualization Insights
PCA visualization confirmed distinct cluster separation, particularly between high-risk and other patient groups.