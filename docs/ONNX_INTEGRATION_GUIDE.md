# HyperGrid ONNX Model Integration Guide
**Untuk React Native & TypeScript / Web**

Dokumen ini berisi panduan lengkap integrasi dan contoh implementasi model Machine Learning **HyperGrid (`hg_model_v1.onnx`)** pada aplikasi sisi *client* (mobile / web).

---

## 1. Spesifikasi Model ONNX

| Properti | Detail |
| :--- | :--- |
| **Nama Berkas** | `hg_model_v1.onnx` |
| **Ukuran Berkas** | **~41.1 KB** (Sangat ringan, cocok untuk mobile) |
| **Tipe Input Tensor** | `float_input` |
| **Dimensi Input** | `[BatchSize, 32]` (Tipe: `float32`) |
| **Output Tensor 1** | `label` $\rightarrow$ `int64` (`0`: Normal, `1`: Pothole) |
| **Output Tensor 2** | `probabilities` $\rightarrow$ `float32` $\rightarrow$ `[P(Normal), P(Pothole)]` |
| **Optimal Decision Threshold** | **`0.47`** (Probabilitas $\ge 0.47$ diklasifikasikan sebagai Pothole) |
| **Dependensi Eksternal** | **TIDAK ADA** (Tanpa Scaler / Encoder terpisah) |

---

## 2. Urutan 32 Fitur Input Tensor

Model mengharapkan array `Float32Array` berisi tepat **32 angka numerik** dalam urutan berikut:

| Indeks | Nama Fitur | Deskripsi |
| :--- | :--- | :--- |
| **0 – 5** | `dynamic_acc_mean`, `std`, `min`, `max`, `range`, `rms` | Statistik akselerasi dinamis (guncangan) |
| **6 – 11** | `gyro_mag_mean`, `std`, `min`, `max`, `range`, `rms` | Statistik magnitudo rotasi giroskop |
| **12 – 17** | `jerk_mean`, `std`, `min`, `max`, `range`, `rms` | Statistik lonjakan getaran ($\Delta \text{dynamic\_acc}$) |
| **18 – 23** | `angular_jerk_mean`, `std`, `min`, `max`, `range`, `rms` | Statistik hentakan putaran ($\Delta \text{gyro\_mag}$) |
| **24 – 29** | `speed_clean_mean`, `std`, `min`, `max`, `range`, `rms` | Statistik kecepatan kendaraan (km/jam) |
| **30** | `dynamic_acc_crest_factor` | $\text{Peak} / \text{RMS}$ (Ketajaman lonjakan getaran) |
| **31** | `dynamic_acc_zcr` | *Zero Crossing Rate* sinyal getaran |

---

## 3. Modul Ekstraksi Fitur (TypeScript)

Simpan kode ini di file `RoadFeatureExtractor.ts`:

```typescript
// RoadFeatureExtractor.ts

export interface SensorSample {
  timestamp: number; // detik atau milidetik
  ax: number;        // accelerometer X
  ay: number;        // accelerometer Y
  az: number;        // accelerometer Z
  gx: number;        // gyro X (rad/s atau deg/s)
  gy: number;        // gyro Y
  gz: number;        // gyro Z
  speed: number;     // kecepatan kendaraan (km/jam)
}

export class RoadFeatureExtractor {
  private windowSize: number;

  constructor(windowSize = 20) {
    this.windowSize = windowSize;
  }

  /**
   * Menghitung statistik dasar: Mean, Std, Min, Max, Range, RMS
   */
  private computeStats(arr: number[]) {
    const n = arr.length;
    if (n === 0) return { mean: 0, std: 0, min: 0, max: 0, range: 0, rms: 0 };

    let sum = 0;
    let sumSq = 0;
    let min = arr[0];
    let max = arr[0];

    for (let i = 0; i < n; i++) {
      const v = arr[i];
      sum += v;
      sumSq += v * v;
      if (v < min) min = v;
      if (v > max) max = v;
    }

    const mean = sum / n;
    const variance = Math.max(0, sumSq / n - mean * mean);
    const std = Math.sqrt(variance);
    const range = max - min;
    const rms = Math.sqrt(sumSq / n);

    return { mean, std, min, max, range, rms };
  }

  /**
   * Mengubah 1 window (20 sampel sensor) menjadi Float32Array panjang 32
   */
  public extractFeatures(windowSamples: SensorSample[]): Float32Array {
    if (windowSamples.length < this.windowSize) {
      throw new Error(`Dibutuhkan minimal ${this.windowSize} sampel.`);
    }

    const samples = windowSamples.slice(-this.windowSize);

    // 1. Hitung Magnitudo Vektor (Orientation-Invariant)
    const accMag = samples.map(s => Math.sqrt(s.ax * s.ax + s.ay * s.ay + s.az * s.az));
    const gyroMag = samples.map(s => Math.sqrt(s.gx * s.gx + s.gy * s.gy + s.gz * s.gz));
    const speedArr = samples.map(s => s.speed || 0.0);

    // 2. Baseline Gravitasi Lokal (Median) & Dynamic Acceleration
    const sortedAcc = [...accMag].sort((a, b) => a - b);
    const medianAcc = sortedAcc[Math.floor(sortedAcc.length / 2)];
    const dynamicAcc = accMag.map(v => Math.abs(v - medianAcc));

    // 3. Jerk (Turunan pertama)
    const jerk: number[] = [0];
    const angularJerk: number[] = [0];
    for (let i = 1; i < dynamicAcc.length; i++) {
      jerk.push(Math.abs(dynamicAcc[i] - dynamicAcc[i - 1]));
      angularJerk.push(Math.abs(gyroMag[i] - gyroMag[i - 1]));
    }

    // 4. Hitung Statistik untuk 5 Sinyal Utama
    const dynStats = this.computeStats(dynamicAcc);
    const gyroStats = this.computeStats(gyroMag);
    const jerkStats = this.computeStats(jerk);
    const angJerkStats = this.computeStats(angularJerk);
    const speedStats = this.computeStats(speedArr);

    // 5. Fitur Sinyal Tambahan (Crest Factor & ZCR)
    const crestFactor = dynStats.max / (dynStats.rms + 1e-6);

    let zeroCrossings = 0;
    for (let i = 1; i < dynamicAcc.length; i++) {
      const prev = dynamicAcc[i - 1] - dynStats.mean;
      const curr = dynamicAcc[i] - dynStats.mean;
      if (prev * curr < 0) zeroCrossings++;
    }
    const zcr = zeroCrossings / (dynamicAcc.length - 1 + 1e-6);

    // 6. Susun tepat 32 elemen Float32Array
    return new Float32Array([
      dynStats.mean, dynStats.std, dynStats.min, dynStats.max, dynStats.range, dynStats.rms,
      gyroStats.mean, gyroStats.std, gyroStats.min, gyroStats.max, gyroStats.range, gyroStats.rms,
      jerkStats.mean, jerkStats.std, jerkStats.min, jerkStats.max, jerkStats.range, jerkStats.rms,
      angJerkStats.mean, angJerkStats.std, angJerkStats.min, angJerkStats.max, angJerkStats.range, angJerkStats.rms,
      speedStats.mean, speedStats.std, speedStats.min, speedStats.max, speedStats.range, speedStats.rms,
      crestFactor,
      zcr,
    ]);
  }
}
```

---

## 4. Contoh Integrasi React Native

### A. Instalasi Dependensi
```bash
npm install onnxruntime-react-native react-native-sensors
# atau dengan yarn:
# yarn add onnxruntime-react-native react-native-sensors
```

### B. Service Deteksi Jalan (`RoadDetectionService.ts`)

```typescript
// RoadDetectionService.ts
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import { RoadFeatureExtractor, SensorSample } from './RoadFeatureExtractor';

export class RoadDetectionService {
  private session: InferenceSession | null = null;
  private extractor: RoadFeatureExtractor;
  private buffer: SensorSample[] = [];
  private readonly WINDOW_SIZE = 20; // 20 sampel (~2 detik pada 10Hz)
  private readonly STRIDE = 5;       // Evaluasi setiap 5 sampel baru
  private readonly OPTIMAL_THRESHOLD = 0.47;

  constructor() {
    this.extractor = new RoadFeatureExtractor(this.WINDOW_SIZE);
  }

  /**
   * Inisialisasi Model ONNX dari Assets / Local Storage
   */
  public async initialize(modelPath: string): Promise<void> {
    try {
      this.session = await InferenceSession.create(modelPath);
      console.log('HyperGrid ONNX Model loaded successfully.');
    } catch (err) {
      console.error('Gagal memuat model ONNX:', err);
      throw err;
    }
  }

  /**
   * Mengalirkan data sensor baru dan melakukan inferensi secara streaming
   */
  public async pushSensorData(
    sample: SensorSample,
    onPotholeDetected?: (confidence: number, peakShock: number) => void
  ): Promise<{ isPothole: boolean; confidence: number } | null> {
    this.buffer.push(sample);

    // Jalankan inferensi jika buffer sudah cukup dan memenuhi interval stride
    if (this.buffer.length >= this.WINDOW_SIZE && this.buffer.length % this.STRIDE === 0) {
      // Pertahankan buffer maksimal 40 sampel agar hemat memori
      if (this.buffer.length > 40) {
        this.buffer = this.buffer.slice(-this.WINDOW_SIZE);
      }

      return await this.runInference(onPotholeDetected);
    }

    return null;
  }

  private async runInference(
    onPotholeDetected?: (confidence: number, peakShock: number) => void
  ): Promise<{ isPothole: boolean; confidence: number }> {
    if (!this.session) throw new Error('Model ONNX belum diinisialisasi.');

    // 1. Ekstraksi 32 Fitur
    const featureVector = this.extractor.extractFeatures(this.buffer);

    // 2. Buat Input Tensor [1, 32]
    const inputTensor = new Tensor('float32', featureVector, [1, 32]);

    // 3. Jalankan ONNX Model
    const feeds: Record<string, Tensor> = { float_input: inputTensor };
    const results = await this.session.run(feeds);

    // 4. Baca Hasil
    const probabilities = results.probabilities.data as Float32Array;
    const potholeConfidence = Number(probabilities[1]); // Probabilitas Pothole
    const isPothole = potholeConfidence >= this.OPTIMAL_THRESHOLD;

    if (isPothole && onPotholeDetected) {
      const peakShock = featureVector[3]; // dynamic_acc_max
      onPotholeDetected(potholeConfidence, peakShock);
    }

    return { isPothole, confidence: potholeConfidence };
  }
}
```

---

## 5. Contoh Komponen React Native (`RoadMonitorComponent.tsx`)

```tsx
// RoadMonitorComponent.tsx
import React, { useEffect, useState, useRef } from 'react';
import { View, Text, StyleSheet, Button, Alert } from 'react-native';
import { accelerometer, gyroscope, setUpdateIntervalForType, SensorTypes } from 'react-native-sensors';
import { RoadDetectionService } from './RoadDetectionService';

// Set sampling rate ke 10 Hz (100ms per update)
setUpdateIntervalForType(SensorTypes.accelerometer, 100);
setUpdateIntervalForType(SensorTypes.gyroscope, 100);

export const RoadMonitorComponent: React.FC = () => {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [status, setStatus] = useState('Standby');
  const [potholeCount, setPotholeCount] = useState(0);
  const detectorRef = useRef<RoadDetectionService>(new RoadDetectionService());

  const currentAcc = useRef({ x: 0, y: 0, z: 0 });
  const currentGyro = useRef({ x: 0, y: 0, z: 0 });

  useEffect(() => {
    // Inisialisasi model ONNX saat komponen dimount
    detectorRef.current.initialize('hg_model_v1.onnx').catch((err) => {
      Alert.alert('Error', 'Gagal memuat model ML');
    });
  }, []);

  useEffect(() => {
    let accSub: any;
    let gyroSub: any;

    if (isMonitoring) {
      accSub = accelerometer.subscribe(({ x, y, z }) => {
        currentAcc.current = { x, y, z };
      });

      gyroSub = gyroscope.subscribe(async ({ x, y, z }) => {
        currentGyro.current = { x, y, z };

        // Masukkan sampel ke detector
        await detectorRef.current.pushSensorData(
          {
            timestamp: Date.now() / 1000,
            ax: currentAcc.current.x,
            ay: currentAcc.current.y,
            az: currentAcc.current.z,
            gx: currentGyro.current.x,
            gy: currentGyro.current.y,
            gz: currentGyro.current.z,
            speed: 35.0, // Kecepatan dari GPS (km/jam)
          },
          (confidence, peakShock) => {
            // Triggered saat terdeteksi lubang!
            setStatus(`⚠️ LUBANG TERDETEKSI! (${(confidence * 100).toFixed(0)}%)`);
            setPotholeCount((prev) => prev + 1);

            // Opsional: Bunyikan haptik/audio atau masukkan payload ke antrean kirim
          }
        );
      });
    }

    return () => {
      accSub?.unsubscribe();
      gyroSub?.unsubscribe();
    };
  }, [isMonitoring]);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>HyperGrid Road Sensor</Text>
      <Text style={styles.status}>{status}</Text>
      <Text style={styles.counter}>Total Lubang: {potholeCount}</Text>

      <Button
        title={isMonitoring ? 'Stop Monitoring' : 'Start Drive'}
        color={isMonitoring ? '#EF4444' : '#009EFF'}
        onPress={() => setIsMonitoring(!isMonitoring)}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#121212' },
  title: { fontSize: 22, fontWeight: 'bold', color: '#F8FAFC', marginBottom: 20 },
  status: { fontSize: 18, color: '#38BDF8', marginBottom: 10 },
  counter: { fontSize: 16, color: '#94A3B8', marginBottom: 30 },
});
```

---

## 6. Integrasi untuk React Web / Node.js (`onnxruntime-web`)

Jika ingin menjalankan simulasi di Web / Dashboard Next.js:

```bash
npm install onnxruntime-web
```

```typescript
import * as ort from 'onnxruntime-web';
import { RoadFeatureExtractor } from './RoadFeatureExtractor';

async function runWebInference(sensorWindow: any[]) {
  // 1. Muat session WebAssembly
  const session = await ort.InferenceSession.create('./models/hg_model_v1.onnx');

  // 2. Ekstrak 32 Fitur
  const extractor = new RoadFeatureExtractor(20);
  const features = extractor.extractFeatures(sensorWindow);

  // 3. Inference
  const tensor = new ort.Tensor('float32', features, [1, 32]);
  const output = await session.run({ float_input: tensor });

  console.log('Hasil Web Prediksi:', output.label.data[0]);
}
```

---

## 7. Tips Optimasi & Best Practices

1. **Hemat Daya Baterai:**
   - Gunakan `STRIDE = 5` atau `10`. Tidak perlu melakukan inferensi pada setiap sampel tunggal sensor. Inferensi 2 kali per detik sudah lebih dari cukup untuk menangkap lubang jalan.
2. **Kirim Data Secara Batch:**
   - Ketika `isPothole === true`, simpan koordinat GPS ke database lokal (SQLite/AsyncStorage). Kirimkan ke server backend secara batch setiap 1–5 menit untuk menghemat kuota internet pengguna.
3. **Over-The-Air (OTA) Updates:**
   - Karena ukuran model hanya **41.1 KB**, Anda dapat menyimpan versi terbaru di cloud storage (Supabase / S3) dan mengunduh model pengganti langsung ke filesystem lokal smartphone tanpa perlu merilis pembaruan di Google Play Store atau App Store.
