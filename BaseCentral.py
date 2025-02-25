import pandas as pd
import os
import json
from pathlib import Path
from collections import defaultdict

class BaseCentral:
    def __init__(self):
        self.valid_option = False
        self.option = None
        self.num = None
        self.df = None
        self.bot = None
        self.notBot = None
        self.Scenarios = {
            1: ["IRC", "SPAM", "HTTP"],
            2: ["IRC", "SPAM", "HTTP"],
            3: ["IRC", "TCP", "HTTP"],
            4: ["IRC", "DNS", "HTTP"],
            5: ["SPAM", "TCP", "HTTP"],
            6: ["TCP"],
            7: ["HTTP"],
            8: ["TCP"],
            9: ["IRC", "SPAM", "HTTP", "TCP"],
            10: ["IRC", "DNS", "HTTP"],
            11: ["IRC", "DNS", "HTTP"],
            12: ["P2P"],
            13: ["SPAM", "TCP", "HTTP"]
        }
        self.df1_list = []
        self.used_indices_file = './database/used_indices.json'
        self.used_indices = self.load_used_indices()

    def load_used_indices(self):
        """Carga los índices usados desde un archivo JSON único"""
        try:
            if Path(self.used_indices_file).exists():
                with open(self.used_indices_file, 'r') as f:
                    data = json.load(f)
                    return defaultdict(set, {int(k): set(v) for k, v in data.items()})
            return defaultdict(set)
        except Exception as e:
            print(f"Error loading indices: {e}")
            return defaultdict(set)

    def save_used_indices(self):
        """Guarda los índices usados en archivo JSON"""
        try:
            with open(self.used_indices_file, 'w') as f:
                serializable = {str(k): list(v) for k, v in self.used_indices.items()}
                json.dump(serializable, f, indent=2)
        except Exception as e:
            print(f"Error saving indices: {e}")

    def Selection(self):
        print("\nPick your option to continue")
        print("-------------------------------------")
        for i in range(1, 14):
            print(f"{i}. Scenario {i}")
        print("-------------------------------------\n")

        while not self.valid_option:
            self.option = input("Pick the scenery you want to use (1-13): ")
            if self.option.isdigit() and 1 <= int(self.option) <= 13:
                self.num = int(self.option)
                file_path = f'./database/{self.num}.binetflow'
                if not Path(file_path).exists():
                    print(f"Error: File {file_path} not found!")
                    continue
                self.df = pd.read_csv(file_path, sep=',')
                self.valid_option = True
            else:
                print("Invalid input. Please enter a number between 1 and 13.")

        # Cargar índices para el escenario actual
        scenario_indices = self.used_indices[self.num]
        total_bots = int(input("Total number of bots you want to use: "))
        self.notBot = int(input("Number of Not bots you want to use: "))

        attack_types = self.Scenarios[self.num]
        num_attacks = len(attack_types)
        bots_per_type = total_bots // num_attacks
        remaining_bots = total_bots % num_attacks

        actual_total_bots = 0
        print(f"\n{' Attack Type ':-^30}")

        for i, attack in enumerate(attack_types):
            current_bots = bots_per_type + 1 if i < remaining_bots else bots_per_type
            
            # Filtrar datos disponibles
            condition = (
                self.df['Label'].str.contains('Botnet', case=False) & 
                self.df['Label'].str.contains(attack, case=False)
            )
            available_data = self.df[condition].drop(index=scenario_indices, errors='ignore')
            available_count = len(available_data)
            
            # Manejar casos con datos insuficientes
            if available_count < current_bots:
                print(f"Warning: Only {available_count} bots available for {attack} (requested {current_bots})")
                current_bots = min(available_count, current_bots)
                print(f"Selected {current_bots} bots for {attack}")
            else:
                print(f"Selected {current_bots} bots for {attack}")
            
            # Seleccionar muestras si hay datos disponibles
            if current_bots > 0 and available_count > 0:
                selected = available_data.sample(n=current_bots)
                self.df1_list.append(selected)
                scenario_indices.update(selected.index)
                actual_total_bots += len(selected)
            elif current_bots > 0:
                print(f"No bots available for {attack} after filtering")

        # Manejar NotBots
        print(f"\n{' Not Bots ':-^30}")
        condition_normal = (
            self.df['Label'].str.contains('Normal', case=False) |
            self.df['Label'].str.contains('Background', case=False)
        )
        available_normal = self.df[condition_normal].drop(index=scenario_indices, errors='ignore')
        available_normal_count = len(available_normal)
        
        if available_normal_count < self.notBot:
            print(f"Warning: Only {available_normal_count} not bots available (requested {self.notBot})")
            self.notBot = min(available_normal_count, self.notBot)
            print(f"Selected {self.notBot} not bots")
        else:
            print(f"Selected {self.notBot} not bots")
        
        if self.notBot > 0 and available_normal_count > 0:
            selected_normal = available_normal.sample(n=self.notBot)
            self.df1_list.append(selected_normal)
            scenario_indices.update(selected_normal.index)

        # Actualizar y guardar índices
        self.used_indices[self.num] = scenario_indices
        self.save_used_indices()

        # Crear dataset final
        if self.df1_list:
            final_df = pd.concat(self.df1_list)
            output_path = './database/0.binetflow'
            
            # Escribir encabezado solo si el archivo no existe
            header = not Path(output_path).exists()
            final_df.to_csv(output_path, mode='a', header=header, index=False)
            
            print(f"\n{' Result ':-^40}")
            print(f"Requested bots: {total_bots}")
            print(f"Actual bots selected: {actual_total_bots}")
            print(f"Total not bots selected: {self.notBot}")
            print(f"Total registros agregados: {actual_total_bots + self.notBot}")
            print(f"Dataset guardado en: {output_path}")
            print(f"Índices actualizados guardados en: {self.used_indices_file}")
        else:
            print("\nError: No se seleccionaron datos válidos!")

if __name__ == '__main__':
    base_central = BaseCentral()
    base_central.Selection()