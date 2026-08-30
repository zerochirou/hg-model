# HyperGrid Model Labs
---

HyperGrid Model Labs merupakan model machine learning yang dirancang untuk mengklasifikasikan kondisi permukaan dan kerusakan jalan berdasarkan data sensor inersial yang dikumpulkan dari accelerometer dan gyroscope.

Model menganalisis enam komponen utama sensor, yaitu Ax, Ay, Az, Gx, Gy, dan Gz, untuk mengidentifikasi pola getaran, percepatan, serta perubahan orientasi yang muncul ketika kendaraan melewati berbagai kondisi jalan.

Untuk meningkatkan representasi data, sinyal sensor dibagi ke dalam sejumlah time window. Pada setiap window, sistem melakukan ekstraksi fitur statistik seperti mean, standard deviation, maximum, minimum, dan Root Mean Square (RMS). Fitur-fitur tersebut kemudian digunakan sebagai masukan ke model XGBoost untuk melakukan klasifikasi multi-kelas.

Arsitektur pipeline HyperGrid terdiri dari:

```text
                    HYPERGRID MODEL LABS
                           │
                           ▼
              Gyroscope + Accelerometer Data
                           │
          ┌────────────────┴────────────────┐
          │                                 │
          ▼                                 ▼
       Gx Gy Gz                         Ax Ay Az
          │                                 │
          └────────────────┬────────────────┘
                           ▼
                    Data Cleaning
                           │
                           ▼
                  File-Based Grouping
                           │
                           ▼
                 Sliding Time Window
              Window: 100 | Step: 50
                           │
                           ▼
                 Statistical Features
          Mean │ Std │ Min │ Max │ RMS
                           │
                           ▼
                  ~30 Features / Window
                           │
                           ▼
                 Group-Aware Train/Test Split
                           │
                           ▼
                       XGBoost
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
      Classification   Confusion      Learning
         Report         Matrix         Curve
```