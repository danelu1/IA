# Descriere set de date

Acest set de date conține informații relevante acordării de împrumuturi de către o instituție 
financiară diferiților aplicanți (persoane fizice). Setul de date conține 8000 de înregistrări și 13 atribute
care descriu aplicanții și decizia de acordare a împrumutului (variabila țintă). Atributele sunt următoarele:

| Nume atribut                    | Valori posibile                                                                     | Descriere                                                      |
|---------------------------------|-------------------------------------------------------------------------------------|----------------------------------------------------------------|
| `applicant_age`                 | Numere întregi pozitive                                                             | Vârsta solicitantului de credit.                               |
| `applicant_income`              | Numere reale pozitive                                                               | Venitul solicitantului de credit.                              |
| `residential_status`            | 'Owner', 'Renter', 'Mortgage', 'Unknown'                                            | Statutul rezidențial al solicitantului de credit.              |
| `job_tenure_years`              | Numere întregi pozitive                                                             | Numărul de ani de activitate pe piața muncii a solicitantului. |
| `loan_purpose`                  | 'Study', 'Business', 'Personal', 'Health', 'Home Improvement', 'Debt Consolidation' | Scopul împrumutului.                                           |
| `loan_rating`                   | 'Excellent', 'Very Good', 'Good', 'Fair', 'Poor', 'Very Poor', 'Extremely Poor'     | Clasificarea împrumutului.                                     |
| `loan_amount`                   | Numere reale pozitive                                                               | Suma împrumutului.                                             |
| `loan_rate`                     | Numere reale pozitive                                                               | Rata dobânzii la împrumut.                                     |
| `loan_income_ratio`             | Numere reale pozitive                                                               | Raportul dintre suma împrumutului și venitul solicitantului.   |
| `credit_history_default_status` | 'Yes', 'No', 'NaN'                                                                  | Starea implicită a istoricului de credit al solicitantului.    |
| `credit_history_length_years`   | Numere întregi pozitive                                                             | Lungimea istoricului de credit al solicitantului în ani.       |
| `credit_history_length_months`  | Numere întregi pozitive                                                             | Lungimea istoricului de credit al solicitantului în luni.      |
| `stability_rating`              | 'A', 'B', 'C', 'D'                                                                  | Stabilitatea financiară a solicitantului de credit.            |
| `loan_approval_status`          | 'Approved', 'Declined'                                                              | **Starea de aprobare a împrumutului (atributul țintă).**       |

Ultima coloană din tabel reprezintă variabila țintă, care indică dacă un aplicant a primit sau nu un împrumut

## Împărțirea setului de date

Setul de date este împărțit în două fișiere: `credit_risk_train.csv` și `credit_risk_test.csv`.
`credit_risk_train.csv` conține 8000 de înregistrări și toate atributele din tabelul de mai sus. 
Acestea vor fi folosite pentru a antrena un model de învățare automată.

`credit_risk_test.csv` conține 2000 de înregistrări și toate atributele din tabelul de mai sus.
Acestea vor fi folosite pentru a testa modelul de învățare automată antrenat pe setul de date de antrenament.

Pe lângă acestea, se mai furnizează fișierul `credit_risk_full.csv`, care conține întregul set de date laolaltă (train + test).
