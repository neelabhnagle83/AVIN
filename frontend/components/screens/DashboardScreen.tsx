import React from 'react';
import { View, Text, ScrollView, TouchableOpacity, StyleSheet, Image } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { styles } from '@/components/styles/DashboardStyles';

export default function DashboardScreen() {
  return (
    <ScrollView style={styles.container}>
      {/* Header Section */}
      <View style={styles.header}>
        <Text style={styles.welcomeText}>Welcome to AVIN,</Text>
        <Text style={styles.username}>Neelabh</Text>
      </View>

      {/* Weather Card */}
      <View style={styles.weatherCard}>
        <View style={styles.weatherTemp}>
          <Text style={styles.temperature}>19°C</Text>
          <Text style={styles.weatherText}>H: 24°C</Text>
        </View>
        <View style={styles.weatherIcon}>
          <Ionicons name="partly-sunny" size={40} color="#686B30" />
        </View>
      </View>

      {/* Notification Card */}
      <View style={styles.notificationCard}>
        <Text style={styles.notificationTitle}>Notify</Text>
        <Text style={styles.notificationText}>as a member of the board.</Text>
        <TouchableOpacity style={styles.joinButton}>
          <Text style={styles.joinButtonText}>Join the board</Text>
        </TouchableOpacity>
        <Text style={styles.notificationSubtext}>to be completed and work with customers.</Text>
      </View>

      {/* My Lands Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>My Lands</Text>
        <Text style={styles.sectionDescription}>Description is useful in order!</Text>
        
        <View style={styles.benefitsContainer}>
          <Text style={styles.benefitsTitle}>Benefits of Adding Land</Text>
          <View style={styles.benefitItem}>
            <Ionicons name="checkmark-circle" size={16} color="#686B30" />
            <Text style={styles.benefitText}>Get processed and make it easier</Text>
          </View>
          <View style={styles.benefitItem}>
            <Ionicons name="checkmark-circle" size={16} color="#686B30" />
            <Text style={styles.benefitText}>Make your land more properly</Text>
          </View>
          <View style={styles.benefitItem}>
            <Ionicons name="checkmark-circle" size={16} color="#686B30" />
            <Text style={styles.benefitText}>Provide good crops with new toys</Text>
          </View>
        </View>
      </View>

      {/* Agri Store Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Agri Store</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.storeScroll}>
          {/* Store Item 1 */}
          <View style={styles.storeItem}>
            <View style={styles.storeItemContent}>
              <Text style={styles.storeItemText}>Append these stores</Text>
              <Text style={styles.storeItemText}>Place them on our own</Text>
              <Text style={styles.storeItemText}>Stop buying</Text>
              <Text style={styles.storeItemText}>Buy them</Text>
              <Text style={styles.storeItemText}>Save money</Text>
              <Text style={styles.storeItemText}>Buy them</Text>
            </View>
          </View>
          
          {/* Store Item 2 (Duplicate for demo) */}
          <View style={styles.storeItem}>
            <View style={styles.storeItemContent}>
              <Text style={styles.storeItemText}>Append these stores</Text>
              <Text style={styles.storeItemText}>Place them on our own</Text>
              <Text style={styles.storeItemText}>Stop buying</Text>
              <Text style={styles.storeItemText}>Buy them</Text>
              <Text style={styles.storeItemText}>Save money</Text>
              <Text style={styles.storeItemText}>Buy them</Text>
            </View>
          </View>
        </ScrollView>
      </View>

      {/* AVIN-AI Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Meet AVIN-AI</Text>
        <View style={styles.aiContainer}>
          <View style={styles.aiItem}>
            <Ionicons name="checkmark-circle" size={16} color="#686B30" />
            <Text style={styles.aiText}>Clean access</Text>
          </View>
          <View style={styles.aiItem}>
            <Ionicons name="checkmark-circle" size={16} color="#686B30" />
            <Text style={styles.aiText}>Clean use</Text>
          </View>
          <View style={styles.aiItem}>
            <Ionicons name="checkmark-circle" size={16} color="#686B30" />
            <Text style={styles.aiText}>Clean access</Text>
          </View>
          <View style={styles.aiItem}>
            <Ionicons name="checkmark-circle" size={16} color="#686B30" />
            <Text style={styles.aiText}>Clean access</Text>
          </View>
          <View style={styles.aiItem}>
            <Ionicons name="checkmark-circle" size={16} color="#686B30" />
            <Text style={styles.aiText}>Clean access</Text>
          </View>
        </View>
      </View>

      {/* Safety Tip */}
      <View style={styles.safetyTip}>
        <Text style={styles.safetyText}>Safety Tip</Text>
        <Text style={styles.safetySubtext}>Only for yourself, please – could be one type of trigger delivery.</Text>
      </View>
    </ScrollView>
  );
}