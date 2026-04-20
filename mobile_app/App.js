import { Ionicons } from "@expo/vector-icons";
import * as ImagePicker from "expo-image-picker";
import { StatusBar } from "expo-status-bar";
import { useMemo, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Image,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";

const DEFAULT_API_URL =
  process.env.EXPO_PUBLIC_API_URL ||
  (Platform.OS === "android" ? "http://10.0.2.2:8000" : "http://localhost:8000");
const MULTIVIEW_ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330];

async function imageToFormPart(image, fallbackName) {
  const name = image.fileName || fallbackName;
  const extension = name.split(".").pop()?.toLowerCase() || "jpg";
  const type = extension === "png" ? "image/png" : "image/jpeg";

  if (Platform.OS === "web") {
    const response = await fetch(image.uri);
    const blob = await response.blob();
    return new File([blob], name, { type: blob.type || type });
  }

  return { uri: image.uri, name, type };
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(digits);
}

export default function App() {
  const [mode, setMode] = useState("quick");
  const [weight, setWeight] = useState("");
  const [topImage, setTopImage] = useState(null);
  const [sideImage, setSideImage] = useState(null);
  const [sideImages, setSideImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const canSubmit = useMemo(() => {
    const parsedWeight = Number(weight);
    if (!parsedWeight || parsedWeight <= 0 || !topImage) return false;
    if (mode === "quick") return Boolean(sideImage);
    return sideImages.length >= 3;
  }, [mode, sideImage, sideImages.length, topImage, weight]);

  function resetInputs() {
    setWeight("");
    setTopImage(null);
    setSideImage(null);
    setSideImages([]);
    setLoading(false);
    setResult(null);
  }

  function switchMode(nextMode) {
    if (nextMode === mode) return;
    resetInputs();
    setMode(nextMode);
  }

  async function requestPermission(source) {
    const permission =
      source === "camera"
        ? await ImagePicker.requestCameraPermissionsAsync()
        : await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert("Permission needed", "Camera or photo access is needed to use this app.");
      return false;
    }
    return true;
  }

  async function pickImage(target, source = "library") {
    const allowed = await requestPermission(source);
    if (!allowed) return;

    const picker =
      source === "camera"
        ? ImagePicker.launchCameraAsync
        : ImagePicker.launchImageLibraryAsync;
    const response = await picker({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: false,
      quality: 0.9,
      allowsMultipleSelection: target === "multi" && source === "library",
      selectionLimit: target === "multi" ? 12 : 1,
    });

    if (response.canceled || !response.assets?.length) return;
    if (target === "top") setTopImage(response.assets[0]);
    if (target === "side") setSideImage(response.assets[0]);
    if (target === "multi") {
      setSideImages((current) => [...current, ...response.assets].slice(0, 12));
    }
    setResult(null);
  }

  async function buildFormData() {
    const form = new FormData();
    form.append("weight_g", String(Number(weight)));
    form.append("top_image", await imageToFormPart(topImage, "top.jpg"));

    if (mode === "quick") {
      form.append("side_image", await imageToFormPart(sideImage, "side.jpg"));
    } else {
      const parts = await Promise.all(
        sideImages.map((image, index) => imageToFormPart(image, `side_${index}.jpg`))
      );
      parts.forEach((part) => form.append("side_images", part));
      form.append("angles_deg", JSON.stringify(MULTIVIEW_ANGLES.slice(0, sideImages.length)));
    }
    return form;
  }

  async function submit() {
    if (!canSubmit) {
      Alert.alert(
        "Missing input",
        mode === "quick"
          ? "Add weight, top photo, and side photo."
          : "Add weight, top photo, and at least 3 side photos."
      );
      return;
    }

    setLoading(true);
    setResult(null);
    try {
      const endpoint = mode === "quick" ? "/estimate/quick" : "/estimate/multiview";
      const response = await fetch(`${DEFAULT_API_URL}${endpoint}`, {
        method: "POST",
        body: await buildFormData(),
        headers: { Accept: "application/json" },
      });
      const body = await response.json();
      if (!response.ok) {
        throw new Error(body.detail || "Estimation failed.");
      }
      setResult(body);
    } catch (error) {
      Alert.alert("Estimation failed", error.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar style="dark" />
      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : undefined}
        style={styles.keyboard}
      >
        <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
          <Text style={styles.title}>Gold Purity Estimator</Text>
          <Text style={styles.subtitle}>
            Place a dark 1 cm square marker beside the object in every photo.
          </Text>

          <View style={styles.segment}>
            <Pressable
              style={[styles.segmentButton, mode === "quick" && styles.segmentActive]}
              onPress={() => switchMode("quick")}
            >
              <Text style={[styles.segmentText, mode === "quick" && styles.segmentTextActive]}>
                Quick
              </Text>
            </Pressable>
            <Pressable
              style={[styles.segmentButton, mode === "multiview" && styles.segmentActive]}
              onPress={() => switchMode("multiview")}
            >
              <Text
                style={[styles.segmentText, mode === "multiview" && styles.segmentTextActive]}
              >
                Multiview
              </Text>
            </Pressable>
          </View>

          <LabeledInput
            label="Weight in grams"
            value={weight}
            onChangeText={setWeight}
            keyboardType="decimal-pad"
            placeholder="Example: 12.8"
          />

          <PhotoSlot
            label="Top photo"
            image={topImage}
            onCamera={() => pickImage("top", "camera")}
            onLibrary={() => pickImage("top", "library")}
          />

          {mode === "quick" ? (
            <PhotoSlot
              label="Side photo"
              image={sideImage}
              onCamera={() => pickImage("side", "camera")}
              onLibrary={() => pickImage("side", "library")}
            />
          ) : (
            <View style={styles.section}>
              <View style={styles.sectionHeader}>
                <Text style={styles.sectionTitle}>Side photos</Text>
                <Text style={styles.counter}>{sideImages.length}/12</Text>
              </View>
              <Text style={styles.helper}>
                Add images in rotation order. The app assigns angles from 0 to 330 degrees.
              </Text>
              <View style={styles.row}>
                <ActionButton
                  icon="camera-outline"
                  label="Take"
                  onPress={() => pickImage("multi", "camera")}
                />
                <ActionButton
                  icon="images-outline"
                  label="Choose"
                  onPress={() => pickImage("multi", "library")}
                />
                <ActionButton
                  icon="trash-outline"
                  label="Clear"
                  tone="muted"
                  onPress={() => {
                    setSideImages([]);
                    setResult(null);
                  }}
                />
              </View>
              <View style={styles.thumbnailGrid}>
                {sideImages.map((image, index) => (
                  <Image key={`${image.uri}-${index}`} source={{ uri: image.uri }} style={styles.thumb} />
                ))}
              </View>
            </View>
          )}

          <Pressable
            style={[styles.submit, (!canSubmit || loading) && styles.submitDisabled]}
            onPress={submit}
            disabled={!canSubmit || loading}
          >
            {loading ? (
              <ActivityIndicator color="#ffffff" />
            ) : (
              <Text style={styles.submitText}>Estimate purity</Text>
            )}
          </Pressable>

          {result && <ResultPanel result={result} />}
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

function LabeledInput({ label, ...props }) {
  return (
    <View style={styles.field}>
      <Text style={styles.label}>{label}</Text>
      <TextInput {...props} style={styles.input} placeholderTextColor="#7a7f87" />
    </View>
  );
}

function ActionButton({ icon, label, onPress, tone = "primary" }) {
  return (
    <Pressable
      style={[styles.actionButton, tone === "muted" && styles.actionButtonMuted]}
      onPress={onPress}
    >
      <Ionicons name={icon} size={18} color={tone === "muted" ? "#2d333b" : "#ffffff"} />
      <Text style={[styles.actionText, tone === "muted" && styles.actionTextMuted]}>{label}</Text>
    </Pressable>
  );
}

function PhotoSlot({ label, image, onCamera, onLibrary }) {
  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{label}</Text>
      {image ? (
        <Image source={{ uri: image.uri }} style={styles.preview} />
      ) : (
        <View style={styles.emptyPreview}>
          <Ionicons name="image-outline" size={30} color="#7a7f87" />
          <Text style={styles.emptyText}>No photo selected</Text>
        </View>
      )}
      <View style={styles.row}>
        <ActionButton icon="camera-outline" label="Take photo" onPress={onCamera} />
        <ActionButton icon="images-outline" label="Choose" onPress={onLibrary} />
      </View>
    </View>
  );
}

function ResultPanel({ result }) {
  const purity = result.purity;
  const volume = result.volume;
  const dimensions = volume.dimensions_cm || {};
  const diagnostics = volume.diagnostics || {};
  const rangeRows = Object.entries(result.karat_density_ranges || {}).sort(
    ([a], [b]) => Number(b) - Number(a)
  );

  return (
    <View style={styles.result}>
      <Text style={styles.resultTitle}>Result</Text>
      <View style={styles.metricRow}>
        <Metric label="Equation estimate" value={`${formatNumber(purity.karat_formula_clamped, 1)}K`} />
        <Metric label="Density" value={`${formatNumber(purity.density_g_cm3, 2)} g/cm3`} />
      </View>
      <View style={styles.metricRow}>
        <Metric label="Volume" value={`${formatNumber(volume.volume_cm3, 2)} cm3`} />
        <Metric
          label="Range estimate"
          value={
            purity.range_match_karat
              ? `${purity.range_match_karat}K`
              : `Nearest ${purity.closest_karat || "-"}K`
          }
        />
      </View>

      <View style={styles.detailBlock}>
        <Text style={styles.detailTitle}>Volume Check</Text>
        {volume.method === "quick_two_view" ? (
          <>
            <Text style={styles.detailText}>
              Top area: {formatNumber(dimensions.top_area_cm2, 3)} cm2
            </Text>
            <Text style={styles.detailText}>
              Weighted side height: {formatNumber(dimensions.avg_height_cm, 3)} cm
            </Text>
            <Text style={styles.detailText}>
              Top scale: {formatNumber(diagnostics.top_scale_cm_per_px, 5)} cm/px
            </Text>
            <Text style={styles.detailText}>
              Side scale: {formatNumber(diagnostics.side_scale_cm_per_px, 5)} cm/px
            </Text>
          </>
        ) : (
          <>
            <Text style={styles.detailText}>Width: {formatNumber(dimensions.width_cm, 3)} cm</Text>
            <Text style={styles.detailText}>Depth: {formatNumber(dimensions.depth_cm, 3)} cm</Text>
            <Text style={styles.detailText}>Height: {formatNumber(dimensions.height_cm, 3)} cm</Text>
          </>
        )}
      </View>

      {volume.visuals?.top_mask && (
        <ImagePanel title="Top Area Mask" uri={volume.visuals.top_mask} />
      )}
      {volume.visuals?.side_height_mask && (
        <ImagePanel title="Side Height Mask" uri={volume.visuals.side_height_mask} />
      )}

      <View style={styles.detailBlock}>
        <Text style={styles.detailTitle}>Purity Methods</Text>
        <Text style={styles.detailText}>Equation: {result.equation}</Text>
        <Text style={styles.detailText}>
          Equation output: {formatNumber(purity.karat_formula, 2)}K
        </Text>
        <Text style={styles.detailText}>Range output: {purity.classification}</Text>
      </View>

      <View style={styles.rangeTable}>
        <Text style={styles.detailTitle}>Density Ranges</Text>
        {rangeRows.map(([karat, limits]) => (
          <View key={karat} style={styles.rangeRow}>
            <Text style={styles.rangeKarat}>{karat}K</Text>
            <Text style={styles.rangeValue}>
              {formatNumber(limits.min, 2)} - {formatNumber(limits.max, 2)} g/cm3
            </Text>
          </View>
        ))}
      </View>

      {result.visuals?.density_graph && (
        <ImagePanel title="Density vs Karat Graph" uri={result.visuals.density_graph} wide />
      )}

      <Text style={styles.classification}>{purity.classification}</Text>
      <Text style={styles.warning}>{result.warning}</Text>
    </View>
  );
}

function ImagePanel({ title, uri, wide = false }) {
  return (
    <View style={styles.imagePanel}>
      <Text style={styles.detailTitle}>{title}</Text>
      <Image
        source={{ uri }}
        style={[styles.resultImage, wide && styles.graphImage]}
        resizeMode="contain"
      />
    </View>
  );
}

function Metric({ label, value }) {
  return (
    <View style={styles.metric}>
      <Text style={styles.metricLabel}>{label}</Text>
      <Text style={styles.metricValue}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: "#f7f8fa",
  },
  keyboard: {
    flex: 1,
  },
  content: {
    padding: 20,
    gap: 16,
  },
  title: {
    color: "#16191d",
    fontSize: 30,
    fontWeight: "800",
    letterSpacing: 0,
  },
  subtitle: {
    color: "#4a515b",
    fontSize: 15,
    lineHeight: 22,
    letterSpacing: 0,
  },
  segment: {
    flexDirection: "row",
    backgroundColor: "#e8ebef",
    borderRadius: 8,
    padding: 4,
  },
  segmentButton: {
    flex: 1,
    minHeight: 44,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 6,
  },
  segmentActive: {
    backgroundColor: "#ffffff",
  },
  segmentText: {
    color: "#4a515b",
    fontWeight: "700",
    letterSpacing: 0,
  },
  segmentTextActive: {
    color: "#16191d",
  },
  field: {
    gap: 8,
  },
  label: {
    color: "#2d333b",
    fontWeight: "700",
    letterSpacing: 0,
  },
  input: {
    minHeight: 48,
    borderWidth: 1,
    borderColor: "#c7ccd3",
    borderRadius: 8,
    paddingHorizontal: 14,
    color: "#16191d",
    backgroundColor: "#ffffff",
    fontSize: 16,
    letterSpacing: 0,
  },
  section: {
    gap: 12,
  },
  sectionHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  sectionTitle: {
    color: "#16191d",
    fontSize: 18,
    fontWeight: "800",
    letterSpacing: 0,
  },
  counter: {
    color: "#4a515b",
    fontWeight: "700",
    letterSpacing: 0,
  },
  helper: {
    color: "#4a515b",
    lineHeight: 20,
    letterSpacing: 0,
  },
  preview: {
    width: "100%",
    height: 210,
    borderRadius: 8,
    backgroundColor: "#dfe3e8",
  },
  emptyPreview: {
    height: 160,
    borderWidth: 1,
    borderColor: "#c7ccd3",
    borderRadius: 8,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#ffffff",
    gap: 8,
  },
  emptyText: {
    color: "#5f6874",
    letterSpacing: 0,
  },
  row: {
    flexDirection: "row",
    gap: 10,
  },
  actionButton: {
    flex: 1,
    minHeight: 44,
    borderRadius: 8,
    backgroundColor: "#1e7f64",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    gap: 6,
  },
  actionButtonMuted: {
    backgroundColor: "#dfe3e8",
  },
  actionText: {
    color: "#ffffff",
    fontWeight: "800",
    letterSpacing: 0,
  },
  actionTextMuted: {
    color: "#2d333b",
  },
  thumbnailGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  thumb: {
    width: 72,
    height: 72,
    borderRadius: 6,
    backgroundColor: "#dfe3e8",
  },
  submit: {
    minHeight: 52,
    borderRadius: 8,
    backgroundColor: "#a86612",
    alignItems: "center",
    justifyContent: "center",
  },
  submitDisabled: {
    opacity: 0.45,
  },
  submitText: {
    color: "#ffffff",
    fontSize: 17,
    fontWeight: "900",
    letterSpacing: 0,
  },
  result: {
    gap: 14,
    backgroundColor: "#ffffff",
    borderColor: "#c7ccd3",
    borderWidth: 1,
    borderRadius: 8,
    padding: 16,
  },
  resultTitle: {
    color: "#16191d",
    fontSize: 22,
    fontWeight: "900",
    letterSpacing: 0,
  },
  metricRow: {
    flexDirection: "row",
    gap: 10,
  },
  metric: {
    flex: 1,
    minHeight: 78,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#d7dce2",
    padding: 12,
    justifyContent: "center",
    backgroundColor: "#fafbfc",
  },
  metricLabel: {
    color: "#5f6874",
    fontSize: 12,
    fontWeight: "800",
    letterSpacing: 0,
    textTransform: "uppercase",
  },
  metricValue: {
    color: "#16191d",
    fontSize: 17,
    fontWeight: "900",
    marginTop: 6,
    letterSpacing: 0,
  },
  classification: {
    color: "#2d333b",
    lineHeight: 21,
    letterSpacing: 0,
  },
  warning: {
    color: "#6d4b1f",
    lineHeight: 20,
    letterSpacing: 0,
  },
  detailBlock: {
    gap: 6,
    borderTopWidth: 1,
    borderTopColor: "#e1e5ea",
    paddingTop: 14,
  },
  detailTitle: {
    color: "#16191d",
    fontSize: 16,
    fontWeight: "900",
    letterSpacing: 0,
  },
  detailText: {
    color: "#2d333b",
    lineHeight: 21,
    letterSpacing: 0,
  },
  imagePanel: {
    gap: 8,
  },
  resultImage: {
    width: "100%",
    height: 260,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#d7dce2",
    backgroundColor: "#111418",
  },
  graphImage: {
    height: 230,
    backgroundColor: "#ffffff",
  },
  rangeTable: {
    gap: 8,
    borderTopWidth: 1,
    borderTopColor: "#e1e5ea",
    paddingTop: 14,
  },
  rangeRow: {
    minHeight: 34,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    borderBottomWidth: 1,
    borderBottomColor: "#eef1f4",
  },
  rangeKarat: {
    color: "#16191d",
    fontWeight: "900",
    letterSpacing: 0,
  },
  rangeValue: {
    color: "#4a515b",
    letterSpacing: 0,
  },
});
