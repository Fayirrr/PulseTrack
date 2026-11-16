
import { db } from "./firebase";
import { collection, getDocs } from "firebase/firestore";

export async function testFirebaseConnection() {
  try {
    await getDocs(collection(db, "connection_test"));
    console.log("Firebase connection successful!");
  } catch (error) {
    console.error("Firebase connection test failed:", error.code, error.message);
  }
}
