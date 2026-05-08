#include <bits/stdc++.h>
using namespace std;

int main()
{
    string key = "MEGABUCK";

    string message;
    cout << "Enter message: ";
    getline(cin, message);

    int col = key.length();
    int message_len = message.length();
    int row;
    string msg = key + message;

    if (msg.length() % col == 0)
    {
        row = msg.length() / col;
    }
    else
    {
        row = (msg.length() / col) + 1;
    }

    char matrix[row][col];

    int k = 0;
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            if (k < msg.length())
            {
                matrix[i][j] = msg[k];
                k++;
            }
            else
            {
                matrix[i][j] = 'X';
            }
        }
    }

    cout << "\nMatrix:" << endl;

    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            cout << matrix[i][j] << "  ";
        }
        cout << endl;
        if (i == 0)
        {
            for (int j = 0; j < col; j++)
            {
                cout << "---";
            }
            cout << endl;
        }
    }

    vector<pair<char, int>> order;
    for (int i = 0; i < col; i++)
    {
        order.push_back({matrix[0][i], i});
    }

    sort(order.begin(), order.end());

    string cipher = "";
    for (auto p : order)
    {
        int c = p.second;
        for (int r = 1; r < row; r++)
        {
            cipher += matrix[r][c];
        }
    }

    cout << "\nCipher: " << cipher << endl;

    // DECRYPTION
    char decryptMatrix[row][col];

    for (int i = 0; i < col; i++)
    {
        decryptMatrix[0][i] = key[i];
    }

    int idx = 0;
    for (auto p : order)
    {
        int c = p.second;
        for (int r = 1; r < row; r++)
        {
            if (idx < cipher.length())
            {
                decryptMatrix[r][c] = cipher[idx++];
            }
            else
            {
                decryptMatrix[r][c] = '\0';
            }
        }
    }

    string decrypted = "";
    for (int i = 1; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            if (decryptMatrix[i][j] != '\0')
                decrypted += decryptMatrix[i][j];
        }
    }
    while (!decrypted.empty() && decrypted.back() == 'X')
    {
        decrypted.pop_back();
    }

    cout << "\nDecrypted message: " << decrypted << endl;
}
