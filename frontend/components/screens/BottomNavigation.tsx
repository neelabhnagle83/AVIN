import React, { useEffect, useState } from 'react';
import { View, TouchableOpacity, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useRoute } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { RootStackParamList } from '@/app/navigation/AppNavigator'; // Adjust path if needed

export default function BottomNavigation() {
  const navigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const route = useRoute();

  const [selected, setSelected] = useState(route.name);

  useEffect(() => {
    // Update selected tab when route changes
    setSelected(route.name);
  }, [route.name]);

  const navItems = [
    { key: 'Dashboard', icon: 'home-outline', label: 'Home' },
    { key: 'Crops', icon: 'leaf-outline', label: 'Crops' },
    { key: 'Scan', icon: 'scan-outline', label: 'Scan' },
    { key: 'Assistant', icon: 'mic-outline', label: 'Assistant' },
    { key: 'Profile', icon: 'person-outline', label: 'Profile' },
  ];

  const handlePress = (key: string) => {
    if (key !== selected) {
      navigation.navigate(key as keyof RootStackParamList);
    }
    setSelected(key);
  };

  return (
    <View style={styles.container}>
      {navItems.map((item) => {
        const isActive = selected === item.key;
        const opacity = isActive ? 1 : 0.6;

        return (
          <TouchableOpacity
            key={item.key}
            style={styles.item}
            onPress={() => handlePress(item.key)}
          >
            <Ionicons name={item.icon as any} size={24} color="#fff" style={{ opacity }} />
            <Text style={[styles.label, { opacity }]}>{item.label}</Text>
          </TouchableOpacity>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    backgroundColor: '#444702',
    justifyContent: 'space-around',
    paddingVertical: 10,
    borderTopColor: '#ccc',
    borderTopWidth: 1,
  },
  item: {
    alignItems: 'center',
  },
  label: {
    fontSize: 12,
    color: '#fff',
  },
});
