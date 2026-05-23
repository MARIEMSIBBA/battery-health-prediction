import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import functools



# ============================================================
# 1. DECORATORS
# ============================================================

def timer(func):
    """Mesure le temps d'exécution d'une fonction"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[{func.__name__}] executé en {end - start:.2f}s")
        return result
    return wrapper

def logger(func):
   
    """Affiche un log avant et après chaque fonction"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Début : {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Fin   : {func.__name__}")
        return result
    return wrapper

# ============================================================
# 2. CLASSE Battery (représente UNE batterie)
# ============================================================

class Battery:
    def __init__(self, battery_id, cycle, soh, rul):
        self.battery_id = battery_id   # ex: "B5"
        self.cycle      = cycle        # numéro du cycle
        self.soh        = soh          # State of Health (%)
        self.rul        = rul          # Remaining Useful Life

    def is_healthy(self):
        """Retourne True si SOH > 80%"""
        return self.soh > 80

    def status(self):
        """Retourne le statut de la batterie"""
        if self.soh >= 90:
            return "Bonne"
        elif self.soh >= 80:
            return "Acceptable"
        else:
            return "Dégradée"

    def __repr__(self):
        return f"Battery({self.battery_id} | Cycle {self.cycle} | SOH={self.soh:.1f}% | {self.status()})"

# ============================================================
# 3. CLASSE BatteryDataset (gère le CSV + ML)
# ============================================================

class BatteryDataset:
    def __init__(self, filepath):
        self.filepath = filepath       # chemin du CSV
        self.df       = None           # dataframe pandas
        self.model    = None           # modèle ML
        self.features = ["cycle", "chI", "chV", "chT",
                         "disI", "disV", "disT", "BCt"]  # colonnes d'entrée
        self.target   = "SOH"          # colonne à prédire
        self.target_rul = "RUL"

    @logger
    @timer
    def load_data(self):
        """Charge le fichier CSV"""
        self.df = pd.read_csv(self.filepath)
        print(f"{len(self.df)} lignes chargées")
        return self.df
        
    @logger
    @timer
    def preprocess(self):
      """Nettoie et normalise les données"""
     
      # 1. Supprimer les valeurs manquantes
      before = len(self.df)
      self.df = self.df.dropna()
      after = len(self.df)
      print(f"Lignes supprimées (NaN) : {before - after}")

      # 2. Normalisation (mise à l'échelle)
      self.scaler = StandardScaler()
      self.df[self.features] = self.scaler.fit_transform(self.df[self.features])
      print(f"Normalisation appliquée sur {len(self.features)} features")

      # 3. Sauvegarde du scaler
      import datetime
      timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
      joblib.dump(self.scaler, f"model/scaler_{timestamp}.pkl")
      print(f"Scaler sauvegardé : model/scaler_{timestamp}.pkl")

    

    @logger
    @timer
    def train_soh_model(self):
        """Entraîne un modèle de régression linéaire"""
        df_clean = self.df[self.features + [self.target]].dropna()
        X = df_clean[self.features]
        y = df_clean[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        mae    = mean_absolute_error(y_test, y_pred)
        r2     = r2_score(y_test, y_pred)
        print(f"MAE={mae:.3f} | R²={r2:.3f}")

        os.makedirs("model", exist_ok=True)
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        joblib.dump(self.model, f"model/battery_model_{timestamp}.pkl")

    @logger
    @timer
    def train_rul_model(self):
      """Modèle pour prédire RUL"""
      df_clean = self.df[self.features + ["RUL"]].dropna()

      X = df_clean[self.features]
      y = df_clean["RUL"]

      X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
       )

      self.rul_model = LinearRegression()
      self.rul_model.fit(X_train, y_train)

      y_pred = self.rul_model.predict(X_test)

      print("RUL MAE:", mean_absolute_error(y_test, y_pred))

      import datetime
      timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

      joblib.dump(self.rul_model, f"model/rul_model_{timestamp}.pkl")

    
    @logger
    @timer
    def train_classification_model(self):
        """Classifie la batterie : Bonne / Acceptable / Dégradée"""
        import datetime                                              
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        df_clean = self.df[self.features + [self.target]].dropna().copy()
        # Crée le label selon SOH
        def label(soh):
            if soh >= 90:   return "Bonne"
            elif soh >= 80: return "Acceptable"
            else:           return "Dégradée"

        df_clean["label"] = df_clean["SOH"].apply(label)

        X = df_clean[self.features]
        y = df_clean["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.classifier = LogisticRegression(max_iter=1000)
        self.classifier.fit(X_train, y_train)

        score = self.classifier.score(X_test, y_test)
        print(f"Accuracy classificateur : {score*100:.1f}%")

        joblib.dump(self.classifier, f"model/battery_classifier_{timestamp}.pkl")
        print("Classificateur sauvegardé dans model/battery_classifier.pkl")

    def predict_soh(self, cycle, chI, chV, chT, disI, disV, disT, BCt):
        """Prédit le SOH pour de nouvelles valeurs"""
        data = pd.DataFrame([[cycle, chI, chV, chT, disI, disV, disT, BCt]],
                            columns=self.features)
        prediction = self.model.predict(data)[0]
        return round(float(prediction), 2)
    def predict_rul(self, cycle, chI, chV, chT, disI, disV, disT, BCt):
        """Prédit la durée de vie restante (RUL)"""
    
        data = pd.DataFrame([[cycle, chI, chV, chT, disI, disV, disT, BCt]],
                        columns=self.features)
        prediction = self.rul_model.predict(data)[0]
        return round(float(prediction), 2)
        
    def predict_label(self, cycle, chI, chV, chT, disI, disV, disT, BCt):
        """Prédit le label (Bonne/Acceptable/Dégradée) — Classification"""
        data = pd.DataFrame([[cycle, chI, chV, chT, disI, disV, disT, BCt]],
                            columns=self.features)
        label = self.classifier.predict(data)[0]
        return label


    def get_battery_objects(self):
        """Retourne une liste d'objets Battery"""
        batteries = []
        for _, row in self.df.iterrows():
            b = Battery(row["battery_id"], row["cycle"],
                        row["SOH"], row["RUL"])
            batteries.append(b)
        return batteries



    @logger
    @timer
    def time_series(self, battery_id="B5"):
        """Analyse temporelle : évolution du SOH cycle par cycle"""
        # Filtre les données pour une batterie spécifique
        df_battery = self.df[self.df["battery_id"] == battery_id].copy()
        df_battery = df_battery.sort_values("cycle")

        # Affiche les 10 premiers cycles
        print(f"\nEvolution SOH pour {battery_id} :")
        print(df_battery[["cycle", "SOH", "RUL"]].head(10).to_string(index=False))

        # Graphique
        plt.figure(figsize=(10, 5))
        plt.plot(df_battery["cycle"], df_battery["SOH"], color="blue", label="SOH réel")
        plt.axhline(y=80, color="red", linestyle="--", label="Seuil critique (80%)")
        plt.title(f"Evolution du SOH - Batterie {battery_id}")
        plt.xlabel("Cycle")
        plt.ylabel("SOH (%)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"soh_{battery_id}.png")
        plt.show()
        print(f"Graphique sauvegardé : soh_{battery_id}.png")

        return df_battery[["cycle", "SOH", "RUL"]]
# ============================================================
# 4. MAIN — teste tout
# ============================================================

if __name__ == "__main__":

    # Charge les données
    dataset = BatteryDataset("Battery_dataset.csv")
    dataset.load_data()
    dataset.preprocess()   

    # Régression
    dataset.train_soh_model()
    dataset.train_rul_model()
    
    # Classification
    dataset.train_classification_model()

    # Test Régression
    soh_predit = dataset.predict_soh(
        cycle=50, chI=1.4, chV=4.2, chT=25.44,
        disI=1.98, disV=3.76, disT=32.50, BCt=1.75
    )
    print(f"\nSOH prédit : {soh_predit}%")
    
     # Test Classification
    label_predit = dataset.predict_label(
        cycle=50, chI=1.4, chV=4.2, chT=25.0,
        disI=1.9, disV=3.5, disT=33.0, BCt=1.97
    )
    print(f"Label prédit (Classification) : {label_predit}")

    # Affiche quelques objets Battery
    #batteries = dataset.get_battery_objects()
    #for b in batteries[:3]:
        #print(b)

   # Time Series
    dataset.time_series("B5")