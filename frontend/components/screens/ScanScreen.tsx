import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  Image,
  ScrollView,
  Alert,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';
import styles from '@/components/styles/ScanStyles';
import BottomNavigation from '@/components/screens/BottomNavigation';
import axios from 'axios';
import { API_BASE_URL } from '../../constants/Config'; // Updated the import path to use a relative path

export default function ScanScreen() {
  const [images, setImages] = useState<string[]>([]);
  const [scanResult, setScanResult] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState<boolean>(false);

  const pickImage = async () => {
    console.log('pickImage function triggered');
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permissionResult.granted) {
      Alert.alert('Permission Denied', 'You need to grant permission to access the media library.');
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images, // Reverted to MediaTypeOptions.Images for compatibility
      quality: 1,
      allowsMultipleSelection: true, // Enable multiple image selection
      selectionLimit: 8, // Limit the number of images to 8
    });

    console.log('ImagePicker result:', result);

    if (!result.canceled && result.assets) {
      const selectedImages = result.assets.map((asset) => asset.uri);
      setImages([...images, ...selectedImages]);
    }
  };

  const takePhoto = async () => {
    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: false,
      quality: 1,
    });

    if (!result.canceled && result.assets.length) {
      if (images.length >= 8) {
        Alert.alert('Maximum 8 images allowed');
        return;
      }
      setImages([...images, result.assets[0].uri]);
    }
  };

  const removeImage = (index: number) => {
    const updated = [...images];
    updated.splice(index, 1);
    setImages(updated);
  };

  const startScanning = async () => {
    if (images.length === 0) {
      Alert.alert('No images selected', 'Please upload or capture images to scan.');
      return;
    }

    setIsScanning(true);
    setScanResult(null);
    setConfidence(null);

    try {
      const formData = new FormData();
      images.forEach((uri, index) => {
        formData.append('images', {
          uri,
          name: `image_${index}.jpg`,
          type: 'image/jpeg',
        } as any); // Use 'as any' to bypass type issues
      });

      const response = await axios.post(`${API_BASE_URL}/disease/scan`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }); // Updated to use the correct endpoint `/disease/scan`

      setScanResult(response.data.disease);
      setConfidence(response.data.confidence);
    } catch (error) {
      console.error('Error during scanning:', error);
      Alert.alert('Scanning failed', 'An error occurred while scanning the images.');
    } finally {
      setIsScanning(false);
    }
  };

  return (
    <View style={{ flex: 1 }}>
      <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
        <Text style={styles.heading}>Scan Your Plant</Text>
        <Text style={styles.subtext}>
          Upload or capture an image to detect plant disease instantly.
        </Text>

        <View style={styles.buttonRow}>
          <TouchableOpacity style={styles.uploadButton} onPress={pickImage}>
            <Text style={styles.uploadText}>Upload photos</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.takePhotoButton} onPress={takePhoto}>
            <Text style={styles.takePhotoText}>Take photos</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.previewBox}>
          <Text style={styles.photoInfo}>
            {images.length} photos selected{' '}
            <Text style={styles.maxSize}>(Max size 30MB)</Text>
          </Text>
          <ScrollView horizontal contentContainerStyle={styles.imageList}>
            {images.map((uri, index) => (
              <View key={index} style={styles.imageWrapper}>
                <Image source={{ uri }} style={styles.imageThumb} />
                <TouchableOpacity
                  style={styles.removeIcon}
                  onPress={() => removeImage(index)}
                >
                  <Ionicons name="close-circle" size={16} color="red" />
                </TouchableOpacity>
              </View>
            ))}
            {images.length < 8 && (
              <TouchableOpacity style={styles.addMoreBtn} onPress={pickImage}>
                <Ionicons name="add" size={24} color="#444702" />
              </TouchableOpacity>
            )}
          </ScrollView>
        </View>

        <View style={styles.scanBox}>
          {isScanning ? (
            <Text style={styles.waitingText}>Scanning in progress...</Text>
          ) : scanResult ? (
            <>
              <Text style={styles.resultText}>Result: {scanResult}</Text>
              <Text style={styles.confidenceText}>Confidence: {confidence}</Text>
            </>
          ) : (
            <Text style={styles.waitingText}>Waiting for scan...</Text>
          )}
          <TouchableOpacity style={styles.scanButton} onPress={startScanning} disabled={isScanning}>
            <Text style={styles.scanButtonText}>{isScanning ? 'Scanning...' : 'Start Scanning'}</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>

      <BottomNavigation />
    </View>
  );
}
