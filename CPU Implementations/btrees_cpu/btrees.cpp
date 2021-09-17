#include <assert.h>
#include <iostream>
#include <vector>

#include "btrees.h"

#define NODE_LENGTH 8

btreeNode *initBtree() {
  btreeNode *btree = (btreeNode *)malloc(sizeof(btreeNode));
  btree->isLeaf = true;
  btree->numKeys = 0;
  return btree;
}

int btreeNodeSearchPos(btreeNode *node, int key) {
  int pos = 0;
  while (pos < node->numKeys && key > node->values[pos]) {
    pos++;
  }
  return pos;
}

int btreeNodeSearchPosUpperBound(btreeNode *node, int key) {
  int pos = 0;
  while (pos < node->numKeys && key >= node->values[pos]) {
    pos++;
  }
  return pos;
}

void btreeNodeInsertKey(btreeNode *node, int key) {
  int pos = btreeNodeSearchPos(node, key);
  for (int i = node->numKeys; i > pos; i--) {
    node->values[i] = node->values[i - 1];
  }
  node->values[pos] = key;
  node->numKeys++;
}

void btreeNodeInsertKeyAndChild(btreeNode *node, int key,
                                btreeNode *childNode) {
  int pos = btreeNodeSearchPos(node, key);
  for (int i = node->numKeys; i > pos; i--) {
    node->values[i] = node->values[i - 1];
    node->pointer[i + 1] = node->pointer[i];
  }
  node->values[pos] = key;
  node->pointer[pos + 1] = childNode;
  node->numKeys++;
}

btreeNode *btreeNodeSplitNode(btreeNode *node, int *median) {
  btreeNode *rightSplitNode = (btreeNode *)malloc(sizeof(btreeNode));
  int medianPos = NODE_LENGTH / 2;
  for (int i = medianPos + 1, j = 0; i < node->numKeys; i++, j++) {
    rightSplitNode->values[j] = node->values[i];
    node->values[i] = 0;
  }
  for (int i = medianPos + 1, j = 0; i < node->numKeys + 1; i++, j++) {
    rightSplitNode->pointer[j] = node->pointer[i];
    node->pointer[i] = nullptr;
  }
  rightSplitNode->numKeys = NODE_LENGTH - (medianPos + 1);
  rightSplitNode->isLeaf = node->isLeaf;
  node->numKeys = medianPos;
  *median = node->values[medianPos];
  node->values[medianPos] = 0;
  return rightSplitNode;
}

btreeNode *btreeInsertHelper(btreeNode *node, int *median, int key) {
  if (node->isLeaf) {
    btreeNodeInsertKey(node, key);
  } else {
    int pos = btreeNodeSearchPos(node, key);
    btreeNode *rightSplitNode =
        btreeInsertHelper(node->pointer[pos], median, key);
    if (rightSplitNode) {
      btreeNodeInsertKeyAndChild(node, *median, rightSplitNode);
    }
  }
  if (node->numKeys == NODE_LENGTH) {
    btreeNode *rightSplitNode = btreeNodeSplitNode(node, median);
    return rightSplitNode;
  } else {
    return nullptr;
  }
}

btreeNode *btreeInsert(btreeNode *root, int key) {
  int *median = (int *)malloc(sizeof(int));
  btreeNode *rightSplitNode = btreeInsertHelper(root, median, key);
  if (rightSplitNode) {
    btreeNode *newRoot = (btreeNode *)malloc(sizeof(btreeNode));
    newRoot->isLeaf = false;
    newRoot->numKeys = 1;
    newRoot->values[0] = *median;
    newRoot->pointer[0] = root;
    newRoot->pointer[1] = rightSplitNode;
    root = newRoot;
  }
  return root;
}

int btreeSearch(btreeNode *node, int key) {
  if (node == nullptr) {
    return 0;
  }
  int pos = btreeNodeSearchPos(node, key);
  if (pos < node->numKeys && node->values[pos] == key) {
    return 1;
  }
  return btreeSearch(node->pointer[pos], key);
}

void btreeDelete(btreeNode *node, int key) {
  if (node == nullptr) {
    std::cout << "Element not found" << std::endl;
    return;
  }
  int pos = btreeNodeSearchPos(node, key);
  if (pos < node->numKeys && node->values[pos] == key && node->isLeaf == true) {
    for (int i = pos; i < node->numKeys - 1; i++) {
      node->values[i] = node->values[i + 1];
    }
    node->values[node->numKeys - 1] = 0;
    node->numKeys--;
  } else if (pos < node->numKeys && node->values[pos] == key &&
             node->isLeaf == false) {
    if (node->pointer[pos]->numKeys > 1) {
      btreeNode *leftNode = node->pointer[pos];
      node->values[pos] = leftNode->values[leftNode->numKeys - 1];
      btreeDelete(leftNode, node->values[pos]);
    } else if (node->pointer[pos + 1]->numKeys > 1) {
      node->values[pos] = node->pointer[pos + 1]->values[0];
      btreeDelete(node->pointer[pos + 1], node->values[pos]);
    } else {
      // Merge left and right node
      btreeNode *leftNode = node->pointer[pos];
      btreeNode *rightNode = node->pointer[pos + 1];
      leftNode->values[1] = node->values[pos];
      leftNode->values[2] = rightNode->values[0];
      leftNode->pointer[2] = rightNode->pointer[0];
      leftNode->pointer[3] = rightNode->pointer[1];
      leftNode->numKeys = 3;
      for (int i = pos; i < node->numKeys - 1; i++) {
        node->values[i] = node->values[i + 1];
      }
      node->values[node->numKeys - 1] = 0;
      for (int i = pos + 1; i < node->numKeys; i++) {
        node->pointer[i] = node->pointer[i + 1];
      }
      node->pointer[node->numKeys] = nullptr;
      node->numKeys--;
      btreeDelete(leftNode, key);
    }
  } else {
    btreeDelete(node->pointer[pos], key);
  }
}

int btreeCount(btreeNode *node, int key) {
  if (node == nullptr) {
    return 0;
  }
  int lowerBound = btreeNodeSearchPos(node, key);
  int upperBound = btreeNodeSearchPosUpperBound(node, key);
  int count = upperBound - lowerBound;
  for (int i = lowerBound; i <= upperBound; i++) {
    count += btreeCount(node->pointer[i], key);
  }
  return count;
}