Certainly! Understanding the steps and processes required for node-level, graph-level, and link-level tasks in both transductive and inductive settings can indeed be complex due to the interdependent nature of graph data. Below is an organized table summarizing the steps needed for each possible combination, along with explanations to clarify each approach.

---

## Overview Table of Tasks and Settings

| **Task**               | **Setting**    | **Data Points**                    | **Dataset Splitting**                                                                                                                                                                                                                                                                                                             | **Steps**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|------------------------|----------------|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Node Classification** | **Transductive** | Nodes within a single graph         | - **Graph**: Single, complete graph available during all phases.<br>- **Labels**: Node labels are split into training, validation, and test sets.<br>- **Edges**: All edges are used during message passing in all phases.                                                                                                                                             | 1. **Training Phase**:<br>&nbsp;&nbsp;- Use the entire graph for message passing.<br>&nbsp;&nbsp;- Train the model using labels of training nodes only.<br>2. **Validation Phase**:<br>&nbsp;&nbsp;- Use the same graph for message passing.<br>&nbsp;&nbsp;- Evaluate the model on validation nodes.<br>3. **Test Phase**:<br>&nbsp;&nbsp;- Use the same graph.<br>&nbsp;&nbsp;- Evaluate the model on test nodes.                                                                             |
| **Node Classification** | **Inductive**    | Nodes within separate graphs/subgraphs | - **Graph**: Split into separate subgraphs or use multiple graphs.<br>- **Labels**: Node labels are split accordingly.<br>- **Edges**: Edges between splits are removed; each split contains independent graphs.                                                                                                                                                       | 1. **Training Phase**:<br>&nbsp;&nbsp;- Use training subgraphs/graphs for message passing.<br>&nbsp;&nbsp;- Train the model using labels of training nodes.<br>2. **Validation Phase**:<br>&nbsp;&nbsp;- Use validation subgraphs/graphs.<br>&nbsp;&nbsp;- Evaluate the model on validation nodes.<br>3. **Test Phase**:<br>&nbsp;&nbsp;- Use test subgraphs/graphs.<br>&nbsp;&nbsp;- Evaluate the model on test nodes.<br>- **Note**: Model must generalize to unseen nodes/graphs. |
| **Graph Classification** | **Inductive**    | Entire graphs                       | - **Graph**: Collection of graphs split into training, validation, and test sets.<br>- **Labels**: Each graph has a label.<br>- **Edges**: Each graph is independent; there are no edges between graphs.                                                                                                                                                                | 1. **Training Phase**:<br>&nbsp;&nbsp;- Train the model on training graphs and their labels.<br>2. **Validation Phase**:<br>&nbsp;&nbsp;- Evaluate the model on validation graphs.<br>3. **Test Phase**:<br>&nbsp;&nbsp;- Evaluate the model on test graphs.<br>- **Note**: Model must generalize to unseen graphs.                                                                                                                                |
| **Link Prediction**     | **Transductive** | Pairs of nodes (edges) in a single graph | - **Graph**: Single graph available during all phases.<br>- **Edges**: Edges are split twice:<br>&nbsp;&nbsp;1. **Edge Types**:<br>&nbsp;&nbsp;&nbsp;&nbsp;- **Message Edges**: Used for GNN message passing.<br>&nbsp;&nbsp;&nbsp;&nbsp;- **Supervision Edges**: Held out for prediction tasks.<br>&nbsp;&nbsp;2. **Dataset Splits**:<br>&nbsp;&nbsp;&nbsp;&nbsp;- Supervision edges split into training, validation, and test sets. | 1. **Assign Edge Types**:<br>&nbsp;&nbsp;- Split edges into message and supervision edges.<br>2. **Split Supervision Edges**:<br>&nbsp;&nbsp;- Split supervision edges into training, validation, and test sets.<br>3. **Training Phase**:<br>&nbsp;&nbsp;- Use graph with message edges only.<br>&nbsp;&nbsp;- Predict training supervision edges.<br>4. **Validation Phase**:<br>&nbsp;&nbsp;- Add training supervision edges back into the graph.<br>&nbsp;&nbsp;- Predict validation edges.<br>5. **Test Phase**:<br>&nbsp;&nbsp;- Add validation supervision edges.<br>&nbsp;&nbsp;- Predict test edges.<br>- **Note**: Gradual inclusion of supervision edges simulates real-world scenarios where more edges become known over time. |
| **Link Prediction**     | **Inductive**    | Pairs of nodes within separate graphs | - **Graph**: Multiple graphs split into training, validation, and test sets.<br>- **Edges**: Each graph contains its own message and supervision edges; no edges between graphs.<br>- **Supervision Edges**: Within each graph, split into training, validation, and test sets.                                                                                         | 1. **Training Phase**:<br>&nbsp;&nbsp;- Use training graphs with their message edges.<br>&nbsp;&nbsp;- Predict supervision edges within training graphs.<br>2. **Validation Phase**:<br>&nbsp;&nbsp;- Use validation graphs.<br>&nbsp;&nbsp;- Predict supervision edges within validation graphs.<br>3. **Test Phase**:<br>&nbsp;&nbsp;- Use test graphs.<br>&nbsp;&nbsp;- Predict supervision edges within test graphs.<br>- **Note**: Model must generalize to unseen graphs.          |

---

## Detailed Steps and Explanations

### Node Classification

#### Transductive Setting

- **Dataset**: Single graph where all nodes are connected as per the original structure.
- **Splitting**:
  - **Nodes** are split into training, validation, and test sets.
  - **Edges** remain intact across all phases.

**Steps**:

1. **Training Phase**:
   - Use the entire graph for GNN message passing.
   - Train the model using only the labels of the training nodes.
   - The GNN aggregates information from all nodes (including unlabeled ones) during message passing.
2. **Validation Phase**:
   - Continue using the full graph.
   - Evaluate the model on validation nodes (labels are withheld during training).
3. **Test Phase**:
   - Use the same full graph.
   - Evaluate the model on test nodes.

**Explanation**:

- In the transductive setting, since all nodes are part of a single graph, the GNN can leverage the entire structure for learning.
- Only the node labels are withheld during validation and testing, not the nodes themselves.
  
#### Inductive Setting

- **Dataset**: The graph is partitioned into separate subgraphs, or multiple graphs are used.
- **Splitting**:
  - The graph is divided into training, validation, and test subgraphs.
  - **Edges** between nodes in different splits are removed.

**Steps**:

1. **Training Phase**:
   - Train the model on training subgraphs using their node labels.
2. **Validation Phase**:
   - Evaluate the model on validation subgraphs.
3. **Test Phase**:
   - Evaluate the model on test subgraphs.

**Explanation**:

- The model must learn to generalize to nodes and subgraphs it hasn't seen during training.
- This setting is more challenging as the GNN cannot rely on the global graph structure.

---

### Graph Classification

#### Inductive Setting (Only applicable)

- **Dataset**: A collection of independent graphs.
- **Splitting**:
  - Graphs are split into training, validation, and test sets.

**Steps**:

1. **Training Phase**:
   - Train the model on the training graphs and their labels.
2. **Validation Phase**:
   - Evaluate the model on validation graphs.
3. **Test Phase**:
   - Evaluate the model on test graphs.

**Explanation**:

- Each graph is an individual data point.
- The model must capture patterns that generalize across different graphs.

---

### Link Prediction

#### Transductive Setting

- **Dataset**: Single graph where edges are both part of the structure and the prediction targets.
- **Splitting**:
  - **Edge Types**:
    - **Message Edges**: Used for GNN message passing.
    - **Supervision Edges**: Held out and used as labels for prediction.
  - **Supervision Edges Splitting**:
    - Supervision edges are further split into training, validation, and test sets.

**Steps**:

1. **Assign Edge Types**:
   - Split the original edges into message edges and supervision edges.
2. **Split Supervision Edges**:
   - Divide supervision edges into training, validation, and test sets.
3. **Training Phase**:
   - Use the graph with only message edges.
   - Train the model to predict the existence of training supervision edges.
4. **Validation Phase**:
   - Include **training supervision edges** back into the graph for message passing.
   - Predict validation supervision edges.
5. **Test Phase**:
   - Include **validation supervision edges** into the graph.
   - Predict test supervision edges.

**Explanation**:

- **Why Gradual Inclusion of Supervision Edges?**
  - After training, the model has knowledge of the training supervision edges.
  - Including them during validation allows the model to leverage this known information.
  - Similarly, including validation edges during testing reflects a scenario where new connections become known over time.
- **Avoiding Data Leakage**:
  - By carefully controlling which edges are included during each phase, we prevent the model from "cheating" by using the supervision edges during message passing.

#### Inductive Setting

- **Dataset**: Multiple independent graphs.
- **Splitting**:
  - Graphs are divided into training, validation, and test sets.
  - Each graph contains its own message and supervision edges.

**Steps**:

1. **Training Phase**:
   - Use training graphs with their message edges.
   - Train the model to predict supervision edges within these graphs.
2. **Validation Phase**:
   - Use validation graphs.
   - Predict supervision edges within validation graphs.
3. **Test Phase**:
   - Use test graphs.
   - Predict supervision edges within test graphs.

**Explanation**:

- The model must generalize to entirely new graphs not seen during training.
- Message passing and predictions are confined within each graph.

---

## Additional Notes

- **Transductive vs. Inductive**:
  - **Transductive**: The model has access to the entire graph structure during all phases. It's suitable when the task is to make predictions within a specific, known graph.
  - **Inductive**: The model must generalize to new, unseen graphs or subgraphs. It's essential when deploying models in dynamic environments where new graphs appear over time.

- **Edge Types in Link Prediction**:
  - **Message Edges**: Form the structure the GNN uses to propagate information.
  - **Supervision Edges**: The targets the model is trained to predict.

- **Data Leakage**:
  - Careful splitting and edge assignment prevent the model from accessing information it shouldn't have during training and evaluation phases.

- **Practical Implementation**:
  - Libraries like **PyTorch Geometric (PyG)** and **GraphGym** provide functionalities to handle these complex dataset splits and setups.

---

By following the steps outlined in this table and explanations, you can correctly set up your datasets and training processes for various graph tasks in both transductive and inductive settings. This structured approach ensures that your GNN models are trained and evaluated properly, leading to reliable and generalizable results.
