#include <iostream>
#include <queue>

using namespace std;

class Node {
public:
    Node *left, *right;
    int data;
};

class BinaryTree {
public:
    Node *insert(Node *, int);
    void bfs(Node *);
};

Node *BinaryTree::insert(Node *root, int data) {
    if (!root) {
        root = new Node;
        root->left = NULL;
        root->right = NULL;
        root->data = data;
        return root;
    }

    queue<Node *> q;
    q.push(root);

    while (!q.empty()) {
        Node *temp = q.front();
        q.pop();

        if (temp->left == NULL) {
            temp->left = new Node;
            temp->left->left = NULL;
            temp->left->right = NULL;
            temp->left->data = data;
            return root;
        } else {
            q.push(temp->left);
        }

        if (temp->right == NULL) {
            temp->right = new Node;
            temp->right->left = NULL;
            temp->right->right = NULL;
            temp->right->data = data;
            return root;
        } else {
            q.push(temp->right);
        }
    }

    return root;
}

void BinaryTree::bfs(Node *head) {
    queue<Node*> q;
    q.push(head);

    while (!q.empty()) {
        int qSize = q.size();
        for (int i = 0; i < qSize; i++) {
            Node* currNode = q.front();
            q.pop();
            cout << currNode->data << " ";

            if (currNode->left)
                q.push(currNode->left);
            if (currNode->right)
                q.push(currNode->right);
        }
    }
}

int main() {
    Node *root = NULL;
    BinaryTree binaryTree;

    // Inserting nodes in a different order
    root = binaryTree.insert(root, 4);
    root = binaryTree.insert(root, 2);
    root = binaryTree.insert(root, 6);
    root = binaryTree.insert(root, 1);
    root = binaryTree.insert(root, 3);
    root = binaryTree.insert(root, 5);
    root = binaryTree.insert(root, 7);

    // Performing BFS
    cout << "Breadth First Search: ";
    binaryTree.bfs(root);

    return 0;
}

// g++ -fopenmp bfs.cpp -o bfs
// then: ./bfs