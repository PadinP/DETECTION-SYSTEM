import pandas as pd
import os

class BaseCentral:
    def __init__(self):  # Cambia 'init' a '__init__'
        self.valid_option = False
        self.option = None
        self.num = None
        self.df = None
        self.bot = None
        self.notBot = None
        # HTTP : sera usado para CF , US y HTTP para estos tipos de ataque revisar tabla pagina oficial de ctu-13
    # PortScan(PS) : TCP
        self.Scenarios = { 1: ["IRC", "SPAM", "HTTP"], 
                            2: ["IRC", "SPAM", "HTTP"], 
                            3: ["IRC", "TCP","HTTP"],
                            4: ["IRC", "DNS","HTTP"],
                            5: ["SPAM", "TCP","HTTP"],
                            6: ["TCP"], 
                            7: ["HTTP"],
                            8: ["TCP"], 
                            9: ["IRC", "SPAM", "HTTP", "TCP"], 
                            10: ["IRC", "DNS","HTTP"], 
                            11: ["IRC", "DNS","HTTP"], 
                            12: ["P2P"], 
                            13: ["SPAM", "TCP","HTTP"]}
        self.df1_list = [] # Lista para almacenar los dataframes df1

    def Selection(self):    
        print("\nPick your option to continue")
        print("-------------------------------------")
        print("1. Scenario 1 ")
        print("2. Scenario 2 ")
        print("3. Scenario 3 ")
        print("4. Scenario 4 ")
        print("5. Scenario 5 ")
        print("6. Scenario 6 ")
        print("7. Scenario 7 ")
        print("8. Scenario 8 ")
        print("9. Scenario 9 ")
        print("10. Scenario 10 ")
        print("11. Scenario 11 ")
        print("12. Scenario 12 ")
        print("13. Scenario 13 ")
        print("------------------------------------\n")

        print('Scenarios range from 1 to 13\n')
        
        while not self.valid_option:
            self.option = input("Pick the scenery you want to use: ")
            if self.option.isdigit():
                self.num = int(self.option)
                if 1 <= self.num <= 13:
                    self.df = pd.read_csv(f'./database/{self.num}.binetflow', sep=',')
                    self.valid_option = True
                else:
                    print("Opción inválida")
            else:
                print("Por favor, introduce un número")

        self.bot = int(input("Number of bots you want to use: ")) 
        self.notBot = int(input("Number of Not bots you want to use: ")) 

# Verifica si la clave es igual al número ingresado por el usuario
        if self.num in self.Scenarios:
            self.bot = int(self.bot /  len(self.Scenarios[self.num]))
            print(f"La clave {self.num} tiene {len(self.Scenarios[self.num])} elementos.")
            for value in self.Scenarios[self.num]: # Itera sobre los valores de la clave
                condicion1 = self.df['Label'].str.contains('Botnet', case=False)  & self.df['Label'].str.contains(value, case=False) 
                print(self.bot)
                df1 = self.df[condicion1] # Filtra el dataframe original con la condición1.
                df1 = df1.iloc[:self.bot] # Selecciona las primeras  filas del dataframe df1 segun el numero que tenga la variable 'bot'.
                self.df1_list.append(df1) # Añade el dataframe df1 a la lista df1_list
        else:
            print("La clave no existe en el diccionario.")

        condicion2 = self.df['Label'].str.contains('Normal', case=False) | self.df['Label'].str.contains('Background', case=False)
        df2 = self.df[condicion2]
        df2 = df2.iloc[:self.notBot] 

        self.df = pd.concat(self.df1_list + [df2])

        if not os.path.isfile('0.binetflow'):
            self.df.to_csv('./database/15.binetflow', index=False,)
        else: 
            self.df.to_csv('./database/15.binetflow', mode='a', header=False, index=False)

if __name__ == '__main__':
    base_central = BaseCentral()
    base_central.Selection()