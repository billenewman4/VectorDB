## Hierarchical Clustering Debug Notes - June 1, 2025

### Current Issues:
1. **Missing `_load_data` method**: The error suggests that the HierarchicalClusterer class is missing a `_load_data` method that's being called somewhere in the code.

2. **NumPy local imports in refinement.py**: Found and fixed issues with local NumPy imports in the ClusterRefiner class in refinement.py. All imports should be at the top of the file for proper scoping.

3. **Data preparation seems to work**: The data preparation stage successfully processes the data and creates the necessary output files, but the pipeline fails at some point after that.

### Next steps:
1. Need to determine where `_load_data` is being called and in which class
2. Check for any other method calls that might be missing in the hierarchical clustering implementation
3. Verify data loading and embedding generation functions

### User questions:
What is this function doing in @hierarchical_pipeline.py?   
    def _merge_config(self, default: Dict[str, Any], override: Dict[str, Any]) -> None:
        """
        Recursively merge configuration dictionaries.
        
        Args:
            default: Default configuration (modified in-place)
            override: Override configuration
        """
        for key, value in override.items():
            if isinstance(value, dict) and key in default and isinstance(default[key], dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
---
## Created Cluster
**Cluster ID:** 0
**Parent ID:** 0
**Size:** 2
**Members:** [0, 33]
**Vectors Shape:** (2, 768)
**Vectors (first 3 shown):**
[[ 0.01646831  0.02873847  0.01367021 ...  0.01286738  0.00176173
  -0.01769877]
 [ 0.00743348  0.09280441 -0.0094158  ...  0.02673251 -0.00138247
  -0.03734804]]
---

---
## Created Cluster
**Cluster ID:** 1
**Parent ID:** 0
**Size:** 2
**Members:** [3, 19]
**Vectors Shape:** (2, 768)
**Vectors (first 3 shown):**
[[ 0.03141343 -0.00750902 -0.00254909 ...  0.05698061 -0.05270756
  -0.03135738]
 [-0.0003145  -0.02058238 -0.01048214 ...  0.01467753 -0.07697227
  -0.02094263]]
---

---
## Created Cluster
**Cluster ID:** 2
**Parent ID:** 0
**Size:** 2
**Members:** [5, 47]
**Vectors Shape:** (2, 768)
**Vectors (first 3 shown):**
[[-0.01638692  0.00853771  0.02658351 ...  0.06136458 -0.01157266
   0.01790971]
 [-0.00863045 -0.01870299  0.00586058 ...  0.02208951  0.00415169
  -0.02415154]]
---

---
## Created Cluster
**Cluster ID:** 3
**Parent ID:** 0
**Size:** 4
**Members:** [6, 8, 9, 25]
**Vectors Shape:** (4, 768)
**Vectors (first 3 shown):**
[[ 2.68000849e-02 -3.51462932e-03  2.60280464e-02 ...  3.51949632e-02
  -5.16364686e-02 -5.26481383e-02]
 [ 1.95923857e-02 -1.48603953e-02  3.47500145e-02 ...  1.68905985e-02
  -3.09547409e-02 -1.05705885e-02]
 [ 9.19287049e-05  3.86916846e-02  1.97368972e-02 ...  2.65408736e-02
  -5.30738123e-02 -1.69493034e-02]]
---

---
## Created Cluster
**Cluster ID:** 4
**Parent ID:** 0
**Size:** 2
**Members:** [7, 22]
**Vectors Shape:** (2, 768)
**Vectors (first 3 shown):**
[[ 6.6362672e-02  1.3396699e-02  9.1905147e-03 ... -9.8378370e-03
  -4.5657076e-02 -1.4950684e-02]
 [ 2.7308561e-02  9.0553984e-02 -1.0523667e-02 ...  2.1593260e-02
  -5.4903585e-02  5.3132859e-05]]
---

---
## Created Cluster
**Cluster ID:** 5
**Parent ID:** 0
**Size:** 4
**Members:** [6, 8, 9, 25]
**Vectors Shape:** (4, 768)
**Vectors (first 3 shown):**
[[ 2.68000849e-02 -3.51462932e-03  2.60280464e-02 ...  3.51949632e-02
  -5.16364686e-02 -5.26481383e-02]
 [ 1.95923857e-02 -1.48603953e-02  3.47500145e-02 ...  1.68905985e-02
  -3.09547409e-02 -1.05705885e-02]
 [ 9.19287049e-05  3.86916846e-02  1.97368972e-02 ...  2.65408736e-02
  -5.30738123e-02 -1.69493034e-02]]
---

---
## Created Cluster
**Cluster ID:** 6
**Parent ID:** 0
**Size:** 4
**Members:** [6, 8, 9, 25]
**Vectors Shape:** (4, 768)
**Vectors (first 3 shown):**
[[ 2.68000849e-02 -3.51462932e-03  2.60280464e-02 ...  3.51949632e-02
  -5.16364686e-02 -5.26481383e-02]
 [ 1.95923857e-02 -1.48603953e-02  3.47500145e-02 ...  1.68905985e-02
  -3.09547409e-02 -1.05705885e-02]
 [ 9.19287049e-05  3.86916846e-02  1.97368972e-02 ...  2.65408736e-02
  -5.30738123e-02 -1.69493034e-02]]
---

---
## Created Cluster
**Cluster ID:** 7
**Parent ID:** 0
**Size:** 2
**Members:** [11, 39]
**Vectors Shape:** (2, 768)
**Vectors (first 3 shown):**
[[-0.00527122 -0.07006703  0.02744411 ...  0.01716598 -0.08710639
  -0.01725767]
 [-0.00019462 -0.11016089 -0.00047976 ...  0.02033703 -0.01594258
  -0.04177892]]
---

---
## Created Cluster
**Cluster ID:** 8
**Parent ID:** 0
**Size:** 3
**Members:** [12, 16, 41]
**Vectors Shape:** (3, 768)
**Vectors (first 3 shown):**
[[ 0.00482122  0.04893096  0.03593471 ...  0.05338955 -0.06587544
   0.01127813]
 [-0.01722157 -0.03673444  0.01688889 ...  0.03191196 -0.08074518
  -0.04432295]
 [ 0.01187793  0.04647739  0.01227546 ...  0.05357939 -0.00640767
  -0.01246025]]
---

---
## Created Cluster
**Cluster ID:** 9
**Parent ID:** 0
**Size:** 8
**Members:** [13, 18, 20, 21, 23, 26, 27, 31]
**Vectors Shape:** (8, 768)
**Vectors (first 3 shown):**
[[ 0.01132739 -0.01207151 -0.00137925 ... -0.00082347 -0.01923175
  -0.04103017]
 [ 0.01253497 -0.06654102  0.01700399 ... -0.00278582 -0.02914208
  -0.03184833]
 [ 0.02485408 -0.04876493  0.01985739 ...  0.02196367 -0.03080215
  -0.00174215]]
---

---
## Created Cluster
**Cluster ID:** 10
**Parent ID:** 0
**Size:** 3
**Members:** [12, 16, 41]
**Vectors Shape:** (3, 768)
**Vectors (first 3 shown):**
[[ 0.00482122  0.04893096  0.03593471 ...  0.05338955 -0.06587544
   0.01127813]
 [-0.01722157 -0.03673444  0.01688889 ...  0.03191196 -0.08074518
  -0.04432295]
 [ 0.01187793  0.04647739  0.01227546 ...  0.05357939 -0.00640767
  -0.01246025]]
---

---
## Created Cluster
**Cluster ID:** 11
**Parent ID:** 0
**Size:** 8
**Members:** [13, 18, 20, 21, 23, 26, 27, 31]
**Vectors Shape:** (8, 768)
**Vectors (first 3 shown):**
[[ 0.01132739 -0.01207151 -0.00137925 ... -0.00082347 -0.01923175
  -0.04103017]
 [ 0.01253497 -0.06654102  0.01700399 ... -0.00278582 -0.02914208
  -0.03184833]
 [ 0.02485408 -0.04876493  0.01985739 ...  0.02196367 -0.03080215
  -0.00174215]]
---

---
## Created Cluster
**Cluster ID:** 12
**Parent ID:** 0
**Size:** 2
**Members:** [3, 19]
**Vectors Shape:** (2, 768)
**Vectors (first 3 shown):**
[[ 0.03141343 -0.00750902 -0.00254909 ...  0.05698061 -0.05270756
  -0.03135738]
 [-0.0003145  -0.02058238 -0.01048214 ...  0.01467753 -0.07697227
  -0.02094263]]
---

---
## Created Cluster
**Cluster ID:** 13
**Parent ID:** 0
**Size:** 8
**Members:** [13, 18, 20, 21, 23, 26, 27, 31]
**Vectors Shape:** (8, 768)
**Vectors (first 3 shown):**
[[ 0.01132739 -0.01207151 -0.00137925 ... -0.00082347 -0.01923175
  -0.04103017]
 [ 0.01253497 -0.06654102  0.01700399 ... -0.00278582 -0.02914208
  -0.03184833]
 [ 0.02485408 -0.04876493  0.01985739 ...  0.02196367 -0.03080215
  -0.00174215]]
---

---
## Created Cluster
**Cluster ID:** 14
**Parent ID:** 0
**Size:** 8
**Members:** [13, 18, 20, 21, 23, 26, 27, 31]
**Vectors Shape:** (8, 768)
**Vectors (first 3 shown):**
[[ 0.01132739 -0.01207151 -0.00137925 ... -0.00082347 -0.01923175
  -0.04103017]
 [ 0.01253497 -0.06654102  0.01700399 ... -0.00278582 -0.02914208
  -0.03184833]
 [ 0.02485408 -0.04876493  0.01985739 ...  0.02196367 -0.03080215
  -0.00174215]]
---

---
## Created Cluster
**Cluster ID:** 15
**Parent ID:** 0
**Size:** 2
**Members:** [7, 22]
**Vectors Shape:** (2, 768)
**Vectors (first 3 shown):**
[[ 6.6362672e-02  1.3396699e-02  9.1905147e-03 ... -9.8378370e-03
  -4.5657076e-02 -1.4950684e-02]
 [ 2.7308561e-02  9.0553984e-02 -1.0523667e-02 ...  2.1593260e-02
  -5.4903585e-02  5.3132859e-05]]
---

---
## Created Cluster
**Cluster ID:** 16
**Parent ID:** 0
**Size:** 8
**Members:** [13, 18, 20, 21, 23, 26, 27, 31]
**Vectors Shape:** (8, 768)
**Vectors (first 3 shown):**
[[ 0.01132739 -0.01207151 -0.00137925 ... -0.00082347 -0.01923175
  -0.04103017]
 [ 0.01253497 -0.06654102  0.01700399 ... -0.00278582 -0.02914208
  -0.03184833]
 [ 0.02485408 -0.04876493  0.01985739 ...  0.02196367 -0.03080215
  -0.00174215]]
---

---
## Created Cluster
**Cluster ID:** 17
**Parent ID:** 0
**Size:** 4
**Members:** [6, 8, 9, 25]
**Vectors Shape:** (4, 768)
**Vectors (first 3 shown):**
[[ 2.68000849e-02 -3.51462932e-03  2.60280464e-02 ...  3.51949632e-02
  -5.16364686e-02 -5.26481383e-02]
 [ 1.95923857e-02 -1.48603953e-02  3.47500145e-02 ...  1.68905985e-02
  -3.09547409e-02 -1.05705885e-02]
 [ 9.19287049e-05  3.86916846e-02  1.97368972e-02 ...  2.65408736e-02
  -5.30738123e-02 -1.69493034e-02]]
---

---
## Created Cluster
**Cluster ID:** 18
**Parent ID:** 0
**Size:** 8
**Members:** [13, 18, 20, 21, 23, 26, 27, 31]
**Vectors Shape:** (8, 768)
**Vectors (first 3 shown):**
[[ 0.01132739 -0.01207151 -0.00137925 ... -0.00082347 -0.01923175
  -0.04103017]
 [ 0.01253497 -0.06654102  0.01700399 ... -0.00278582 -0.02914208
  -0.03184833]
 [ 0.02485408 -0.04876493  0.01985739 ...  0.02196367 -0.03080215
  -0.00174215]]
---

---
## Created Cluster
**Cluster ID:** 19
**Parent ID:** 0
**Size:** 8
**Members:** [13, 18, 20, 21, 23, 26, 27, 31]
**Vectors Shape:** (8, 768)
**Vectors (first 3 shown):**
[[ 0.01132739 -0.01207151 -0.00137925 ... -0.00082347 -0.01923175
  -0.04103017]
 [ 0.01253497 -0.06654102  0.01700399 ... -0.00278582 -0.02914208
  -0.03184833]
 [ 0.02485408 -0.04876493  0.01985739 ...  0.02196367 -0.03080215
  -0.00174215]]
---

---
## Created Cluster
**Cluster ID:** 20
**Parent ID:** 0
**Size:** 2
**Members:** [29, 38]
**Vectors Shape:** (2, 768)
**Vectors (first 3 shown):**
[[ 0.01104305  0.06699477  0.02380264 ...  0.05577869 -0.04588069
  -0.00537154]
 [-0.01610474  0.0482767   0.019426   ...  0.03887415 -0.03182772
  -0.00908635]]
---

---
## Created Cluster
**Cluster ID:** 21
**Parent ID:** 0
**Size:** 8
**Members:** [13, 18, 20, 21, 23, 26, 27, 31]
**Vectors Shape:** (8, 768)
**Vectors (first 3 shown):**
[[ 0.01132739 -0.01207151 -0.00137925 ... -0.00082347 -0.01923175
  -0.04103017]
 [ 0.01253497 -0.06654102  0.01700399 ... -0.00278582 -0.02914208
  -0.03184833]
 [ 0.02485408 -0.04876493  0.01985739 ...  0.02196367 -0.03080215
  -0.00174215]]
---

---
## Created Cluster
**Cluster ID:** 22
**Parent ID:** 0
**Size:** 2
**Members:** [0, 33]
**Vectors Shape:** (2, 768)
**Vectors (first 3 shown):**
[[ 0.01646831  0.02873847  0.01367021 ...  0.01286738  0.00176173
  -0.01769877]
 [ 0.00743348  0.09280441 -0.0094158  ...  0.02673251 -0.00138247
  -0.03734804]]
---

---
## Created Cluster
**Cluster ID:** 23
**Parent ID:** 0
**Size:** 2
**Members:** [29, 38]
**Vectors Shape:** (2, 768)
**Vectors (first 3 shown):**
[[ 0.01104305  0.06699477  0.02380264 ...  0.05577869 -0.04588069
  -0.00537154]
 [-0.01610474  0.0482767   0.019426   ...  0.03887415 -0.03182772
  -0.00908635]]
---

---
## Created Cluster
**Cluster ID:** 24
**Parent ID:** 0
**Size:** 2
**Members:** [11, 39]
**Vectors Shape:** (2, 768)
**Vectors (first 3 shown):**
[[-0.00527122 -0.07006703  0.02744411 ...  0.01716598 -0.08710639
  -0.01725767]
 [-0.00019462 -0.11016089 -0.00047976 ...  0.02033703 -0.01594258
  -0.04177892]]
---

---
## Created Cluster
**Cluster ID:** 25
**Parent ID:** 0
**Size:** 3
**Members:** [12, 16, 41]
**Vectors Shape:** (3, 768)
**Vectors (first 3 shown):**
[[ 0.00482122  0.04893096  0.03593471 ...  0.05338955 -0.06587544
   0.01127813]
 [-0.01722157 -0.03673444  0.01688889 ...  0.03191196 -0.08074518
  -0.04432295]
 [ 0.01187793  0.04647739  0.01227546 ...  0.05357939 -0.00640767
  -0.01246025]]
---

---
## Created Cluster
**Cluster ID:** 26
**Parent ID:** 0
**Size:** 2
**Members:** [44, 46]
**Vectors Shape:** (2, 768)
**Vectors (first 3 shown):**
[[ 0.02307442  0.00519195 -0.026868   ...  0.00289709 -0.0124168
   0.00661604]
 [ 0.00152237 -0.03191171 -0.02033973 ... -0.00437414 -0.02066527
  -0.0340138 ]]
---

---
## Created Cluster
**Cluster ID:** 27
**Parent ID:** 0
**Size:** 2
**Members:** [44, 46]
**Vectors Shape:** (2, 768)
**Vectors (first 3 shown):**
[[ 0.02307442  0.00519195 -0.026868   ...  0.00289709 -0.0124168
   0.00661604]
 [ 0.00152237 -0.03191171 -0.02033973 ... -0.00437414 -0.02066527
  -0.0340138 ]]
---

---
## Created Cluster
**Cluster ID:** 28
**Parent ID:** 0
**Size:** 2
**Members:** [5, 47]
**Vectors Shape:** (2, 768)
**Vectors (first 3 shown):**
[[-0.01638692  0.00853771  0.02658351 ...  0.06136458 -0.01157266
   0.01790971]
 [-0.00863045 -0.01870299  0.00586058 ...  0.02208951  0.00415169
  -0.02415154]]
---

---
## Created Cluster
**Cluster ID:** 0
**Parent ID:** 38
**Size:** 2
**Members:** [58, 149]
**Vectors Shape:** (2, 768)
**Vectors (first 3 shown):**
[[ 0.053166    0.03923114  0.01886128 ...  0.02121289 -0.00033982
  -0.03397415]
 [ 0.02504659  0.03049101  0.00632728 ...  0.03812376  0.01061211
  -0.02129298]]
---

---
## Created Cluster
**Cluster ID:** 1
**Parent ID:** 38
**Size:** 2
**Members:** [70, 80]
**Vectors Shape:** (2, 768)
**Vectors (first 3 shown):**
[[ 0.05404195  0.01517002 -0.0057311  ...  0.09871099 -0.04403317
  -0.04607154]
 [ 0.03111934 -0.04671064  0.00519007 ...  0.0736284  -0.02105713
  -0.03579132]]
---

---
## Created Cluster
**Cluster ID:** 2
**Parent ID:** 38
**Size:** 2
**Members:** [78, 150]
**Vectors Shape:** (2, 768)
**Vectors (first 3 shown):**
[[ 0.03927802 -0.02449578  0.02643652 ...  0.03851255 -0.03882921
  -0.02372032]
 [ 0.04332128 -0.04525432  0.02607924 ...  0.02665037 -0.00469982
  -0.02447409]]
---

---
## Created Cluster
**Cluster ID:** 3
**Parent ID:** 38
**Size:** 2
**Members:** [70, 80]
**Vectors Shape:** (2, 768)
**Vectors (first 3 shown):**
[[ 0.05404195  0.01517002 -0.0057311  ...  0.09871099 -0.04403317
  -0.04607154]
 [ 0.03111934 -0.04671064  0.00519007 ...  0.0736284  -0.02105713
  -0.03579132]]
---

---
## Created Cluster
**Cluster ID:** 4
**Parent ID:** 38
**Size:** 2
**Members:** [58, 149]
**Vectors Shape:** (2, 768)
**Vectors (first 3 shown):**
[[ 0.053166    0.03923114  0.01886128 ...  0.02121289 -0.00033982
  -0.03397415]
 [ 0.02504659  0.03049101  0.00632728 ...  0.03812376  0.01061211
  -0.02129298]]
---

---
## Created Cluster
**Cluster ID:** 5
**Parent ID:** 38
**Size:** 2
**Members:** [78, 150]
**Vectors Shape:** (2, 768)
**Vectors (first 3 shown):**
[[ 0.03927802 -0.02449578  0.02643652 ...  0.03851255 -0.03882921
  -0.02372032]
 [ 0.04332128 -0.04525432  0.02607924 ...  0.02665037 -0.00469982
  -0.02447409]]
---
