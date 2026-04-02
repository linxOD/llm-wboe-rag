from pydantic import BaseModel, ValidationError, Field


class Beispiele(BaseModel):
    
    beispielsatz: str = Field(..., description="Ein Beispielsatz, der die Bedeutung laut Belegen veranschaulicht.")
    herkunftsregionen: list[str] = Field(..., description="Die Regionen, aus denen die Belege stammen, die den Beispielsatz unterstützen.")
    beleg_ids: list[str] = Field(..., description="Auflistung von allen relevanten Beleg IDs, aus denen sich der Beispielsatz ableiten lässt.")


class Bedeutung(BaseModel):

    bedeutungskategorie: str = Field(..., description="Die Kategorie der Bedeutung, die aus den Belegen abgeleitet werden kann und eine übergeordnete Kategorie oder Klassifikation darstellt.")
    bedeutungsebene: str = Field(..., description="Die Ebene der Bedeutung, die aus den Belegen abgeleitet werden kann und die Hierarchie oder Stufe der Bedeutung innerhalb der Bedeutungsstruktur darstellt.")
    bedeutung: str = Field(..., description="Die Bedeutung des Wortes, die aus den Belegen abgeleitet werden kann und eine grundlegende Bedeutung oder Verwendung darstellt.")
    regionen: list[str] | None = Field(..., description="Die Regionen, in denen die Bedeutung laut Belegen gilt. Wenn keine spezifischen Regionen vorhanden sind, wird None verwendet.")
    beispiele: list[Beispiele] | None = Field(..., description="Beispielsätze und deren Herkunftsregionen sowie Beleg IDs, die die Bedeutung laut Belegen veranschaulichen. Wenn keine Beispielsätze vorhanden sind, wird None verwendet.")
    beleg_ids: list[str] = Field(..., description="Auflistung von allen relevanten Beleg ID, aus denen die Bedeutung abgeleitet wurden.")


class Artikel(BaseModel):

    lemma: str = Field(..., description="Das Lemma des Wortes.")
    pos: str = Field(..., description="Die Wortart des Wortes (z.B. Substantiv, Verb).")
    genus: str | None = Field(..., description="Der Genus des Wortes, falls relevant oder null.")
    # kasus: str | None = Field(..., description="Der Kasus des Wortes, falls relevant oder null.")
    # diminutiv: str | None = Field(..., description="Der Diminutiv des Wortes, falls relevant oder null.")
    bedeutungen: list[Bedeutung] = Field(..., description="Liste der Bedeutungen, die aus den Belegen abgeleitet werden können und mit dem Wort verbunden sind.")


if __name__ == "__main__":
    json_data = Artikel.model_dump_json()
    try:
        core_meanings = Artikel.model_validate_json(json_data)
        print(core_meanings)
    except ValidationError as e:
        print("Validation error:", e)
