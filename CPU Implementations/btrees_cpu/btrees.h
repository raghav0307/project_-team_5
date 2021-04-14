#ifndef _BTREES_H
#define _BTREES_H

#define NODE_LENGTH 8

struct btreeNode {
  int values[NODE_LENGTH];
  btreeNode *pointer[NODE_LENGTH + 1];
  bool isLeaf;
  int numKeys;
};

btreeNode *initBtree();
int btreeNodeSearchPos(btreeNode *node, int key);
int btreeNodeSearchPosUpperBound(btreeNode *node, int key);
void btreeNodeInsertKey(btreeNode *node, int key);
void btreeNodeInsertKeyAndChild(btreeNode *node, int key, btreeNode *childNode);
btreeNode *btreeNodeSplitNode(btreeNode *node, int *median);
btreeNode *btreeInsertHelper(btreeNode *node, int *median, int key);
btreeNode *btreeInsert(btreeNode *root, int key);
int btreeSearch(btreeNode *node, int key);
void btreeDelete(btreeNode *node, int key);
int btreeCount(btreeNode *node, int key);

#endif
