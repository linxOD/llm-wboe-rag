# from __future__ import annotations
from pydantic import BaseModel, ValidationError, Field
from typing import Literal


class Belegsatz(BaseModel):
    
    ID: str = Field(..., description="Die ID des Belegs.")
    Belegsatz: str = Field(..., description="Der Belegsatz, der im Beleg angegeben ist.")
    Herkunftsort: str = Field(..., description="Im Beleg angegebener Herkunftsort. Wenn kein Herkunftsort vorhanden ist, gib die Großregion an. Wenn beide Angaben leer sind, gib nichts an.")


class Bedeutung(BaseModel):

    Bedeutungsphrase: str = Field(..., description="Die Bedeutung des Wortes, die aus den Belegen abgeleitet werden kann und eine grundlegende Bedeutung oder Verwendung darstellt.")
    Typ: Literal["Hauptbedeutung", "Unterbedeutung", "Bedeutungsvariante"] = Field(..., description="Der Typ der Bedeutung: Hauptbedeutung, Unterbedeutung, Bedeutungsvariante.")
    Gruppierung: Literal[
        "Bedeutung",
        "differenzierte Bedeutung",
        "weitere Bedeutung 1",
        "weitere Bedeutung 2",
        "weitere Bedeutung 3",
        "weitere Bedeutung 4",
        "weitere Bedeutung 5",
        "weitere Bedeutung 6",
        "weitere Bedeutung 7",
        "weitere Bedeutung 8",
        "weitere Bedeutung 9",
        "weitere Bedeutung 10",
        "weitere Bedeutung 11",
    ] = Field(..., description="Ordne zusammenhängede Beudeutungen, Unterbedeutungen und Bedeutungsvarianten den vorgegebenen Gruppierungen zu. Jede Gruppe kann nur einmal verwendet werden, um die Zusammengehörigkeit der Bedeutungen zu kennzeichnen. Die zentralste und häufigste Bedeutung erhält die Gruppierung 'Bedeutung', die nächsthäufigere und zentralste Bedeutung erhält die Gruppierung 'differenzierte Bedeutung' und alle weiteren Bedeutungen erhalten die Gruppierungen 'weitere Bedeutung 1' bis 'weitere Bedeutung 11'.")
    Regionen: list[str] = Field(..., description="Die Großregionen, in denen die Bedeutung laut Belegen gilt.")
    Belegsaetze: list[Belegsatz] = Field(..., description="Ein oder mehere Belegsätze deren Herkunftsregionen sowie ID des Belegs, die die Bedeutung veranschaulichen.")
    Belege: list[str] = Field(..., description="Auflistung sämtlicher relevanten ID(s) der Belege, aus denen die Bedeutung abgeleitet wurden.")


# class Bedeutungen(BaseModel):
    
#     Bedeutungsphrase: str = Field(..., description="Präzise Umschreibung der Bedeutung in eigenen Worten.")
#     Originalformulierungen: list[str] = Field(..., description="Alle in den Belegen vorkommenden Formulierungen dieser Bedeutung (wörtlich zitiert).")
#     Regionen: list[str] = Field(..., description="Alle Großregionen, in denen die Bedeutung belegt ist, mit Angabe der Beleganzahl pro Region.")
#     Belegsaetze: list[Belegsatz] = Field(..., description="Alle aussagekräftigen Belegsätze mit Herkunftsort (nur aus den Belegen, keine eigenen Formulierungen).")
#     Belege: list[str] = Field(..., description="Auflistung sämtlicher ID(s) der Belege, die dieser Bedeutung zugeordnet sind.")
#     Belegsanzahl: int


class Artikel(BaseModel):
    "Main output type for the RAG query generation task prompt3."

    Lemma: str = Field(..., description="Das Lemma der Sammlung.")
    POS: str = Field(..., description="Die Wortart des Lemma (z.B. Substantiv, Verb).")
    Genus: str = Field(..., description="Der Genus des Lemma, falls relevant oder null.")
    Kasus: str = Field(..., description="Der Kasus des Lemma, falls relevant oder null.")
    Bedeutungen: list[Bedeutung] = Field(..., description="Liste der Bedeutungen, die aus den Belegen abgeleitet werden können und mit dem Lemma verbunden sind.")


# class ListeAllerBedeutungen(BaseModel):
#     "Main output type for the RAG query generation task prompt2."
    
#     Bedeutung: list[Bedeutungen] = Field(..., description="Auflistung aller Bedeutungen, die aus den Belegen abzuleiten sind.")
#     RelevanteBelege: list[str] = Field(..., description="Auflistung aller relevanten ID(s) der Belege, die den generierten Bedeutungen zuzuordnen sind.")
#     IrrelevanteBelege: list[str] = Field(..., description="Auflistung aller irrelevanten ID(s) der Belege, die nicht verwendet wurden.")
#     Anmerkungens: str = Field(..., description="Allgemeine Anmerkungen.")


# class Glossar(BaseModel):
#     "Main output type for the RAG query generation task prompt1."
    
#     struktur: list[str] = Field(..., description="Übersicht der verfügbaren Datenkategorien.")
#     belegqualitaet: list[str] = Field(..., description="Bewertung der Vollständigkeit und Konsistenz.")
#     relevante_kategorien: dict[str, str] = Field(..., description="Auflistung der lexikographisch verwertbaren Felder und deren Beschreibung.")
#     probleme: list[str] = Field(..., description="Dokumentation von Datenproblemen.")


if __name__ == "__main__":
    # Create a minimal instance and dump it to JSON —
    # `model_dump_json` is an instance method, not a class method.
    sample = Artikel(lemma="BEISPIEL", pos="Substantiv", genus=None, kasus=None, bedeutung=[])
    json_data = sample.model_dump_json()
    try:
        core_meanings = Artikel.model_validate_json(json_data)
        print(core_meanings)
    except ValidationError as e:
        print("Validation error:", e)
