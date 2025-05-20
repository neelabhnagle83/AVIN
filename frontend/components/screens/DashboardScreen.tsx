// DashboardScreen.tsx
import React, { useState } from 'react';
import { View, Text, ScrollView, TouchableOpacity, Image } from 'react-native';
import { Ionicons, FontAwesome5, MaterialIcons } from '@expo/vector-icons';
import { styles } from '../styles/DashboardStyles';
import BottomNavigation from './BottomNavigation';

export default function DashboardScreen() {
  const [quantities, setQuantities] = useState([0, 0, 0, 0]);

  const handleQuantityChange = (index: number, change: number) => {
    setQuantities(prev => {
      const updated = [...prev];
      updated[index] = Math.max(0, updated[index] + change);
      return updated;
    });
  };

  const currentDate = new Date().toLocaleDateString();

  const products = [
    {
      name: 'Hybrid Tomato Seeds',
      price: '₹120/packet(50g)',
      type: 'Seeds',
      image: require('@/assets/images/store_tomato.png'),
    },
    {
      name: 'Organic Khaad',
      price: '₹150/10kg',
      type: 'Organic Fertilizer',
      image: require('@/assets/images/store_khaad.png'),
    },
    {
      name: 'Urea Fertilizer',
      price: '₹90/5kg',
      type: 'Fertilizer',
      image: require('@/assets/images/store_urea.png'),
    },
    {
      name: 'High-Quality Paddy Seeds',
      price: '₹200/1kg',
      type: 'Seeds',
      image: require('@/assets/images/store_paddy.png'),
    },
  ];

  return (
    <View style={{ flex: 1 }}>
      <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.welcome}>
            Welcome to <Text style={styles.bold}>AVIN,</Text>
          </Text>
          <Text style={styles.subHeader}>Neelabh!</Text>
        </View>

        {/* Weather */}
        <View style={styles.weatherCard}>
          <Text style={styles.dateText}>{currentDate}</Text>
          <View style={{ flexDirection: 'row', alignItems: 'center', gap: 12 }}>
            <Image
              source={require('@/assets/images/cloud-moon.png')}
              style={styles.weatherImage}
            />
            <Text style={styles.weatherDegree}>19°C</Text>
          </View>
          <Text style={styles.feelsLike}>Feels like 24°C</Text>
        </View>

        {/* Profile */}
        <View style={styles.profileCard}>
          <View style={styles.profileRow}>
            <Image source={require('@/assets/images/profile.png')} style={styles.profileIcon} />
            <Text style={styles.profileText}>
              Hi Neelabh{"\n"}Let’s complete your profile in few minutes
            </Text>
          </View>
          <View style={styles.progressBarBackground}>
            <View style={[styles.progressBarFill, { width: '45%' }]} />
          </View>
          <TouchableOpacity style={styles.profileButton}>
            <Text style={styles.profileButtonText}>Complete Profile</Text>
          </TouchableOpacity>
        </View>

        {/* Lands */}
        <View style={styles.landsHeader}>
          <Text style={styles.landsTitle}>My Lands</Text>
          <Ionicons name="add-circle" size={24} color="#3A3502" />
        </View>
        <Text style={styles.noLandText}>Currently no Lands is added</Text>

        {/* Benefits */}
        <View style={styles.benefitsCard}>
          <Image source={require('@/assets/images/farmer.png')} style={styles.farmerImage} />
          <View style={styles.benefitsList}>
            {[
              'Get personalized crop & disease suggestions',
              'Track your land-wise progress',
              'Protect your crops with smart tips',
            ].map((text, i) => (
              <View key={i} style={styles.benefitItem}>
                <MaterialIcons name="check" size={16} color="#3A3502" />
                <Text style={styles.benefitText}>{text}</Text>
              </View>
            ))}
          </View>
        </View>

        {/* Agri Store */}
        <View style={styles.storeHeader}>
          <Text style={styles.sectionTitle}>Agri-Store</Text>
          <TouchableOpacity>
            <Text style={styles.viewMore}>view more</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.storeGrid}>
          {products.map((item, index) => (
            <View key={index} style={styles.productCard}>
              <Image source={item.image} style={styles.productImage} />
              <Text style={styles.productTitle}>{item.name}</Text>
              <Text style={styles.productPrice}>Price {item.price}</Text>
              <Text style={styles.productType}>Type {item.type}</Text>
              <View style={styles.productActions}>
                <TouchableOpacity onPress={() => handleQuantityChange(index, -1)}>
                  <Ionicons name="remove-circle-outline" size={20} color="#3A3502" />
                </TouchableOpacity>
                <Text style={{ marginHorizontal: 8 }}>{quantities[index]}</Text>
                <TouchableOpacity onPress={() => handleQuantityChange(index, 1)}>
                  <Ionicons name="add-circle-outline" size={20} color="#3A3502" />
                </TouchableOpacity>
              </View>
            </View>
          ))}
        </View>

        {/* AVIN AI */}
        <Text style={styles.sectionTitle}>Meet AVIN-AI</Text>
        <View style={styles.aiCard}>
          {[
            'Voice Assistance',
            'Crop Support',
            'Disease Detection Help',
            'Fertilizer & Water Tips',
            '24/7 Smart Chat',
          ].map((feature, i) => (
            <View key={i} style={styles.aiFeature}>
              <FontAwesome5 name="check-circle" size={14} color="#3A3502" />
              <Text style={styles.aiText}>{i + 1}. {feature}</Text>
            </View>
          ))}
          <TouchableOpacity style={styles.askNowButton}>
            <Text style={styles.askNowText}>Ask Now</Text>
          </TouchableOpacity>
        </View>

        {/* Tip Box */}
        <View style={styles.tipBox}>
          <Text style={styles.tipLabel}>Today’s Tip:</Text>
          <Text style={styles.tipText}>Check for yellowing leaves — could be early sign of nitrogen deficiency.</Text>
        </View>
      </ScrollView>

      <BottomNavigation />
    </View>
  );
}
