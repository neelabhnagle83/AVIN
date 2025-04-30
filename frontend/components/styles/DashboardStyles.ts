import { StyleSheet } from 'react-native';

export const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    padding: 16,
  },
  header: {
    marginTop: 48,
    marginBottom: 20,
  },
  welcomeText: {
    fontSize: 24,
    fontWeight: '300',
    color: '#333',
  },
  username: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#686B30',
  },
  weatherCard: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 20,
    marginBottom: 16,
    elevation: 2,
  },
  weatherTemp: {
    flexDirection: 'column',
  },
  temperature: {
    fontSize: 36,
    fontWeight: 'bold',
    color: '#333',
  },
  weatherText: {
    fontSize: 16,
    color: '#666',
  },
  weatherIcon: {
    padding: 10,
  },
  notificationCard: {
    backgroundColor: '#686B30',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
  },
  notificationTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 4,
  },
  notificationText: {
    fontSize: 16,
    color: '#fff',
    marginBottom: 8,
  },
  joinButton: {
    backgroundColor: '#fff',
    borderRadius: 8,
    paddingVertical: 8,
    paddingHorizontal: 16,
    alignSelf: 'flex-start',
    marginBottom: 8,
  },
  joinButtonText: {
    color: '#686B30',
    fontWeight: 'bold',
  },
  notificationSubtext: {
    fontSize: 14,
    color: '#fff',
    opacity: 0.8,
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  sectionDescription: {
    fontSize: 16,
    color: '#666',
    marginBottom: 12,
  },
  benefitsContainer: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
  },
  benefitsTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  benefitItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  benefitText: {
    fontSize: 16,
    color: '#333',
    marginLeft: 8,
  },
  storeScroll: {
    marginHorizontal: -16,
  },
  storeItem: {
    width: 200,
    backgroundColor: '#fff',
    borderRadius: 12,
    marginRight: 16,
    padding: 16,
  },
  storeItemContent: {
    flex: 1,
  },
  storeItemText: {
    fontSize: 16,
    color: '#333',
    marginBottom: 8,
  },
  aiContainer: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
  },
  aiItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  aiText: {
    fontSize: 16,
    color: '#333',
    marginLeft: 8,
  },
  safetyTip: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
  },
  safetyText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginBottom: 4,
  },
  safetySubtext: {
    fontSize: 16,
    color: '#666',
  },
});