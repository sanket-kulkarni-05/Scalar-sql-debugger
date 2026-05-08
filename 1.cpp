#include <bits/stdc++.h>
#include <conio.h>
using namespace std;

// Function to take masked password input
string hidePassword() {
    string password;
    char ch;
    while ((ch = _getch()) != 13) 
    { 
        if (ch == 8) { // Backspace
            if (!password.empty()) {
                password.pop_back();
                cout << "\b \b";
            }
        } else {
            password += ch;
            cout << '*';
        }
    }
    cout << endl;
    return password;
}

// Login function with 3 trials
int login(string username, string def_password, string input_username, string input_password) {
    int trial = 1;
    int status = 0;

    while (trial <= 3) {
        if (username == input_username && def_password == input_password) {
            cout << "Credentials are correct!" << endl;
            cout<<"Entered password: "<<input_password<<endl;
            status = 1;
            break;
        } else {
            if (trial == 3) {
                cout << "Username or password is wrong. Try again after some time." << endl;
                status = 0;
                break;
            }
            cout << "Username or password is wrong, Try again!" << endl;

            cout << "Username: ";
            cin >> input_username;
            cout << "Password: ";
            input_password = hidePassword();

            trial++;
        }
    }
    return status;
}

int main() {
    string username = "sanket123";
    string def_password = "pass@123";

    string input_username, input_password;
    string new_password;

    cout << "Enter username and password for verification" << endl;
    cout << "Username: ";
    cin >> input_username;
    cout << "Password: ";
    input_password = hidePassword();

    int stat = login(username, def_password, input_username, input_password);

    if (stat == 1) {
        cout << "\nEnter New Password: ";
        new_password = hidePassword();
        def_password = new_password;

        cout << "Password reset successfully!" << endl;
        cout<<"New password: "<<def_password<<endl;

        // Verify again
        cout << "\nRe-login with new password" << endl;
        cout << "Username: ";
        cin >> input_username;
        cout << "Password: ";
        input_password = hidePassword();

        login(username, def_password, input_username, input_password);
    }

    return 0;
}
