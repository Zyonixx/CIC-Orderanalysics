from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts.chat import ChatPromptTemplate
from vector import bestell_vector_store, ausgang_vector_store, df_bestell, df_ausgang

model = OllamaLLM(
    model="mistral",
    temperature=0.3,  # Weniger kreative, fokussiertere Antworten
    num_ctx=4096  # Größerer Kontextfenster
)

template = """
Du bist ein Analyse-Bot. Nutze ausschließlich die folgenden Daten:
- Bestellhistorie: {bestellHist}
- Ausgangshistorie: {ausgangHist}
Beantworte die Frage präzise und mit Zahlen. Wenn die Frage nach einer Zählung fragt (z. B. "Wie viele Eilbestellungen?"), zähle die relevanten Einträge korrekt. Gib Empfehlungen, wenn sinnvoll.
Frage: {quest}
"""

def analyze_inventory(df_bestell, df_ausgang):
    recommendations = []
    for artikelnummer in df_bestell["Artikelnummer"].unique():
        bestell_menge = df_bestell[df_bestell["Artikelnummer"] == artikelnummer]["Menge"].sum()
        ausgang_menge = df_ausgang[df_ausgang["Artikelnummer"] == artikelnummer]["VerbrauchteMenge"].sum()
        aktueller_bestand = df_ausgang[df_ausgang["Artikelnummer"] == artikelnummer]["LagerbestandNach"].iloc[-1] if artikelnummer in df_ausgang["Artikelnummer"].values else bestell_menge
        if aktueller_bestand < 0.2 * bestell_menge:  # Schwellenwert: 20% des Bestellvolumens
            recommendations.append(f"Nachbestellung empfohlen für {artikelnummer}: Aktueller Bestand ({aktueller_bestand}) ist niedrig.")
    return recommendations

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("--------------------------------------------")
    quest = input("Programm: Was möchtest du den ChatBot fragen? (q to exit) \n\n\n Du: ")
    manufacturer = input("Gib einen Hersteller an, falls gewünscht (oder lasse es leer): ").strip()
    print("\n\n")
    if quest == "q":
        break
    
    # Erstelle den Filter, falls ein Hersteller angegeben wurde
    filter = {"Lieferant": manufacturer.lower().replace(" gmbh", "").strip()} if manufacturer else None
    
    # Suche mit Filter (falls vorhanden) und einem hohen k-Wert, um alle relevanten Dokumente zu erfassen
    bestell_data = bestell_vector_store.similarity_search(quest, k=1000, filter=filter)
    ausgang_data = ausgang_vector_store.similarity_search(quest, k=1000, filter=filter)
    
    if not bestell_data and not ausgang_data:
        print("Keine relevanten Daten gefunden. Bitte überprüfe deine Frage.")
        continue
    
    recommendations = analyze_inventory(df_bestell, df_ausgang)
    result = chain.invoke({
        "bestellHist": bestell_data,
        "ausgangHist": ausgang_data,
        "quest": quest + "\nEmpfehlungen: " + "\n".join(recommendations)
    })
    print(result)