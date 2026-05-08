#include <bits/stdc++.h>
using namespace std;

int main()
{
    string message, encrypted = "", decrypted = "";
    int k;

    cout << "Enter message: ";
    getline(cin, message);

    cout << "Enter key (k): ";
    cin >> k;
    if (k == 3)
    {
        cout << "This is Traditional cipher" << endl;
    }

    // Encryption
    k = k % 26;

    for (int i = 0; i < message.length(); i++)
    {
        char ch = message[i];

        if (isupper(ch))
        {
            encrypted += char((ch - 'A' + k) % 26 + 'A');
        }
        else if (islower(ch))
        {
            encrypted += char((ch - 'a' + k) % 26 + 'a');
        }
        else
        {
            encrypted += ch;
        }
    }

    // Decryption
    for (int i = 0; i < encrypted.length(); i++)
    {
        char ch = encrypted[i];

        if (isupper(ch))
        {
            decrypted += char((ch - 'A' - k + 26) % 26 + 'A');
        }
        else if (islower(ch))
        {
            decrypted += char((ch - 'a' - k + 26) % 26 + 'a');
        }
        else
        {
            decrypted += ch;
        }
    }

    cout << "\nEncrypted Text: " << encrypted << endl;
    cout << "Decrypted Text: " << decrypted << endl;

    return 0;
}
