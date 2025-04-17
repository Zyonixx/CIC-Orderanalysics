from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import pandas as pd

# Lade die Daten (angenommen, die DataFrames sind bereits definiert)
df_bestell = pd.read_csv("Datenbank/In/bestellhistorie.csv")  # Beispiel-Pfad
df_ausgang = pd.read_csv("Datenbank/Out/ausgangshistorie.csv")  # Beispiel-Pfad

# Initialisiere Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Erstelle Dokumente und Vektorspeicher
add_documents = True  # Beispiel-Bedingung

if add_documents:
    bestell_documents = []
    bestell_ids = []
    for i, row in df_bestell.iterrows():
        try:
            bestell_row = df_bestell[df_bestell["BestellID"] == row["BestellID"]].iloc[0]
            lieferant = bestell_row["Lieferant"]
        except IndexError:
            lieferant = "Unbekannt"
        bestell_document = Document(
            page_content=f"""
            BestellID: {row["BestellID"]}
            Bestelldatum: {row["Bestelldatum"]}
            Bestellart: {row["Bestellart"]}
            Lieferant: {row["Lieferant"]}
            Artikelnummer: {row["Artikelnummer"]}
            Artikelbeschreibung: {row["Artikelbeschreibung"]}
            Menge: {row["Menge"]}
            Einheit: {row["Einheit"]}
            PreisProEinheit: {row["PreisProEinheit"]}
            Bestellstatus: {row["Bestellstatus"]}
            """,
            metadata={"Lieferant": row["Lieferant"].lower().replace(" gmbh", "").strip(), "BestellID": row["BestellID"], "Artikelnummer": row["Artikelnummer"]},
            id=str(i)
        )
        bestell_ids.append(str(i))
        bestell_documents.append(bestell_document)

    ausgang_documents = []
    ausgang_ids = []
    for i, row in df_ausgang.iterrows():
        try:
            bestell_row = df_bestell[df_bestell["BestellID"] == row["BestellID"]].iloc[0]
            lieferant = bestell_row["Lieferant"]
        except IndexError:
            lieferant = "Unbekannt"
        
        ausgang_document = Document(
            page_content=f"""
            AusgangsID: {row["AusgangsID"]}
            Ausgangsdatum: {row["Ausgangsdatum"]}
            BestellID: {row["BestellID"]}
            Artikelnummer: {row["Artikelnummer"]}
            VerbrauchteMenge: {row["VerbrauchteMenge"]}
            LagerbestandVor: {row["LagerbestandVor"]}
            LagerbestandNach: {row["LagerbestandNach"]}
            Bemerkungen: {row["Bemerkungen"]}
            """,
            metadata={
                "Lieferant": lieferant.lower().replace(" gmbh", "").strip(),
                "AusgangsID": row["AusgangsID"],
                "Artikelnummer": row["Artikelnummer"]
            },
            id=f"ausgang_{i}"
        )
    ausgang_ids.append(f"ausgang_{i}")
    ausgang_documents.append(ausgang_document)

    # Erstelle Vektorspeicher
    bestell_vector_store = Chroma.from_documents(
        documents=bestell_documents,
        embedding=embeddings,
        collection_name="bestell_collection",
        persist_directory="./chroma_db_bestell"
    )
    ausgang_vector_store = Chroma.from_documents(
        documents=ausgang_documents,
        embedding=embeddings,
        collection_name="ausgang_collection",
        persist_directory="./chroma_db_ausgang"
    )
else:
    bestell_vector_store = Chroma(
        collection_name="bestell_collection",
        persist_directory="./chroma_db_bestell",
        embedding_function=embeddings
    )
    ausgang_vector_store = Chroma(
        collection_name="ausgang_collection",
        persist_directory="./chroma_db_ausgang",
        embedding_function=embeddings
    )

# Exportiere die Vektorspeicher für main.py
__all__ = ["bestell_vector_store", "ausgang_vector_store", "df_bestell", "df_ausgang"]