# -*- coding: utf-8 -*-
"""
ortak.py — Muğla Orman Yangını Duyarlılığı Projesi: Ortak Veri Hazırlama ve Değerlendirme Modülü.

Tüm notebook'lar bu modülü kullanır; böylece her algoritma AYNI veri, AYNI eğitim/test
bölünmesi ve AYNI metriklerle değerlendirilir (adil karşılaştırma).

Veri kaynağı : veri/orman_yangini.csv  (_builder/veri_hazirla.py üretir)
               Eriş (2026), Mendeley Data, doi:10.17632/mjz29dh3g8.1, CC BY 4.0
Kapsam       : Muğla ili; 2015–2024 yangın kayıtları (OGM Muğla Orman Bölge Müdürlüğü)
Birim        : bir satır = bir örnek nokta (yangın olan ya da olmayan)
Hedef        : Yangin (1 = yangın noktası, 0 = yangın olmayan nokta)
Tekrar üretilebilirlik: tüm rastgeleliklerde random_state = 42.

İki görev
---------
Veri setindeki hedef ikilidir. Kitabın bütün algoritmaları aynı iki görevle sınanır:
  (i)  Regresyon: her noktanın 0–1 arasında bir YANGIN DUYARLILIK SKORU tahmin edilir.
       Hedef 0/1 olduğu için skor, yangın olasılığının bir tahminidir; duyarlılık
       haritaları bu skordan çizilir. Bu görevde R², Brier beceri skoruna eşittir:
       R² = 1 − Brier(model) / Brier(ortalama tahmini). ROC-AUC ise skorun yangın
       noktalarını yangın olmayanlardan ne kadar iyi sıraladığını ölçer.
  (ii) Sınıflandırma: her nokta "Yangın" ya da "Yangın Yok" olarak etiketlenir.

Bölünme hakkında not
--------------------
Ana bölünme, alanyazındaki yangın duyarlılığı çalışmalarının çoğunda olduğu gibi
sınıfa göre katmanlı rastgele %80/%20 bölünmedir. Ancak veri setinde koordinat yoktur
ve birbirine yakın noktalar neredeyse aynı iklim, yükseklik ve arazi değerlerini taşır.
Rastgele bölünme bir test noktasının komşularını eğitime koyar; sonuçlar bu nedenle
iyimserdir. Bu iyimserliği ölçmek için güvenilirlik sınaması SÖZDE MEKÂNSAL grup
katlamalı doğrulamayla yapılır (bkz. cv_katlari): aynı iklim grubundaki noktalar
(yaklaşık aynı bölge) hep birlikte ya eğitimde ya da sınamada kalır.

Özellik seçimi hakkında not
---------------------------
Kaynaktaki 11 etkenin tamamı kullanılır. Bakı (9 sınıf) döngüsel olarak (sin, cos ve
düz göstergesi), arazi örtüsü ise Orman referanslı göstergelerle kodlanır. İklim grubu
(Iklim_Grubu) yalnızca doğrulamada kullanılır, modele girdi olarak verilmez.
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
                             accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix)

RANDOM_STATE = 42
SEED = 42
np.random.seed(SEED)

_KOK = os.path.dirname(os.path.abspath(__file__))
VERI_YOLU = os.path.abspath(os.path.join(_KOK, "..", "veri", "orman_yangini.csv"))
VERI_XLSX = os.path.abspath(os.path.join(_KOK, "..", "veri", "orman_yangini.xlsx"))
TEMIZLIK_YOLU = os.path.abspath(os.path.join(_KOK, "..", "veri", "temizlik_ozeti.json"))
SEKIL_DIR = os.path.abspath(os.path.join(_KOK, "..", "sekiller"))
SONUC_DIR = os.path.abspath(os.path.join(_KOK, "..", "sonuclar"))
os.makedirs(SEKIL_DIR, exist_ok=True)
os.makedirs(SONUC_DIR, exist_ok=True)

TOPOGRAFYA = ["Yukseklik", "Egim", "Baki_Sin", "Baki_Cos", "Baki_Duz"]
IKLIM = ["Sicaklik", "Yagis", "Ruzgar_Hizi"]
BITKI = ["NDVI", "Arazi_Tarim", "Arazi_Calilik", "Arazi_Yerlesim", "Arazi_Su_Ciplak"]
INSAN = ["Yol_Uzaklik", "Yerlesim_Uzaklik", "Akarsu_Uzaklik"]
OZELLIKLER = TOPOGRAFYA + IKLIM + BITKI + INSAN
SAYISAL = ["Yukseklik", "Egim", "Sicaklik", "Yagis", "Ruzgar_Hizi", "NDVI",
           "Yol_Uzaklik", "Yerlesim_Uzaklik", "Akarsu_Uzaklik"]

# Grafiklerde kullanılan okunur adlar
OKUNUR = {
    "Yukseklik": "Yükseklik", "Egim": "Eğim", "Baki_Sin": "Bakı (doğu bileşeni)",
    "Baki_Cos": "Bakı (kuzey bileşeni)", "Baki_Duz": "Düz alan", "Sicaklik": "Sıcaklık",
    "Yagis": "Yağış", "Ruzgar_Hizi": "Rüzgâr hızı", "NDVI": "NDVI",
    "Arazi_Tarim": "Arazi: tarım", "Arazi_Calilik": "Arazi: çalılık/çayır",
    "Arazi_Yerlesim": "Arazi: yerleşim", "Arazi_Su_Ciplak": "Arazi: su/çıplak",
    "Yol_Uzaklik": "Yola uzaklık", "Yerlesim_Uzaklik": "Yerleşime uzaklık",
    "Akarsu_Uzaklik": "Akarsuya uzaklık", "Yangin": "Yangın",
}
BIRIM = {"Yukseklik": "m", "Egim": "°", "Sicaklik": "°C", "Yagis": "mm", "Ruzgar_Hizi": "m/s",
         "NDVI": "", "Yol_Uzaklik": "m", "Yerlesim_Uzaklik": "m", "Akarsu_Uzaklik": "m"}

HEDEF = "Yangin"
GRUP = "Iklim_Grubu"
TEST_ORANI = 0.20
CV_KAT_SAYISI = 5

SINIF_ETIKETLERI = ["Yangın Yok", "Yangın"]
SINIF_ACIKLAMA = ["Yangın Yok (0)", "Yangın (1)"]
POZITIF = "Yangın"

# SVM'nin RBF çekirdeği eğitim süresi örnek sayısının karesiyle büyür. Bu veri setinde
# eğitim kümesi küçük olduğundan alt örnekleme gerekmez; sınır, yalnızca çok büyük veri
# setlerinde devreye girecek bir güvenlik payıdır.
SVM_ORNEK = 8000

KAYNAK = ("Eriş, B. (2026). Forest fire and non-fire sample points with eleven conditioning "
          "factors for Muğla, Türkiye (2015–2024). Mendeley Data, V1")
KAYNAK_DOI = "10.17632/mjz29dh3g8.1"
KAYNAK_URL = "https://data.mendeley.com/datasets/mjz29dh3g8/1"

# Şekiller kitapta en fazla 16 cm genişliğe küçültülür; yazıların basılı sayfada okunur
# kalması için temel yazı boyutları matplotlib varsayılanının üstünde tutulur.
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11.5,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
})

# Şekillerdeki eksen sayıları Türkçe yazımla (ondalık virgül) gösterilir.
import locale as _locale
for _yerel in ("tr_TR.UTF-8", "Turkish_Turkey.1254", "tr_TR"):
    try:
        _locale.setlocale(_locale.LC_NUMERIC, _yerel)
        plt.rcParams["axes.formatter.use_locale"] = True
        break
    except _locale.Error:
        continue


def sayi(x, ondalik=3, isaret=False):
    """Şekil ve metin içi sayılar için Türkçe biçim: 0,728 · −0,31 csv· +0,05."""
    s = f"{x:+.{ondalik}f}" if isaret else f"{x:.{ondalik}f}"
    return s.replace(".", ",").replace("-", "−")


def tam(x):
    """Tam sayı, Türkçe binlik ayırıcıyla: 17636 -> 17.636."""
    return f"{int(round(x)):,}".replace(",", ".")


def veri_yukle(goster=True):
    """Veri setini okur ve sınıf etiketlerini üretir."""
    df = pd.read_csv(VERI_YOLU)
    df["Sinif"] = pd.Categorical(np.where(df[HEDEF] == 1, "Yangın", "Yangın Yok"),
                                 categories=SINIF_ETIKETLERI)
    if goster:
        print(f"{len(df)} nokta · yangın {int(df[HEDEF].sum())} · yangın yok {int((1 - df[HEDEF]).sum())}")
    return df


def egitim_maskesi(df):
    """Sınıfa göre katmanlı, sabit tohumlu %80 eğitim / %20 test bölünmesi."""
    tr, _ = train_test_split(df.index, test_size=TEST_ORANI, stratify=df[HEDEF],
                             random_state=RANDOM_STATE)
    return df.index.isin(tr)


def bolunme(df):
    """Tüm modellerde aynı olan bölünme. Regresyon hedefi 0/1 sayısal, sınıflandırma
    hedefi kategorik etikettir."""
    m = egitim_maskesi(df)
    return (df.loc[m, OZELLIKLER], df.loc[~m, OZELLIKLER],
            df.loc[m, HEDEF].astype(float), df.loc[~m, HEDEF].astype(float),
            df.loc[m, "Sinif"], df.loc[~m, "Sinif"])


def grup_katlari(df, k=CV_KAT_SAYISI):
    """Sözde mekânsal katlar: aynı iklim grubundaki noktalar aynı katta kalır; katlar
    sınıf oranına göre dengelenir. (eğitim_indisleri, sınama_indisleri) üretir."""
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    for a, b in skf.split(df[OZELLIKLER], df[HEDEF], df[GRUP]):
        yield df.index[a], df.index[b]


def cv_katlari(df):
    """Sözde mekânsal grup katlamalı doğrulama. Yalnızca EĞİTİM kümesi kullanılır;
    test kümesine hiç dokunulmaz. Her kat, bazı iklim gruplarını (bölgeleri) bütünüyle
    dışarıda bırakır ve model o bölgelerde sınanır."""
    e = df[egitim_maskesi(df)]
    for i, (a, b) in enumerate(grup_katlari(e), start=1):
        yield (i, e.loc[a, OZELLIKLER], e.loc[b, OZELLIKLER],
               e.loc[a, HEDEF].astype(float), e.loc[b, HEDEF].astype(float),
               e.loc[a, "Sinif"], e.loc[b, "Sinif"])


def olcekli(X_tr, X_te):
    """Mesafe/marjin tabanlı modeller (KNN, SVM, Lojistik) için standartlaştırma."""
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te)


def alt_ornek(X, *diziler, n=SVM_ORNEK, tohum=RANDOM_STATE):
    """Eğitim kümesinden sabit tohumla en fazla n satırlık rastgele alt örneklem seçer."""
    n = min(n, len(X))
    secim = np.random.RandomState(tohum).choice(len(X), size=n, replace=False)
    secim.sort()
    return [d.iloc[secim] if hasattr(d, "iloc") else np.asarray(d)[secim]
            for d in (X,) + diziler]


# Büyüğü iyi olan metrikler (diğerleri hata ölçüsüdür, küçüğü iyidir)
BUYUK_IYI = {"R2", "AUC", "Accuracy", "BalancedAcc", "F1_macro", "F1_weighted"}


def reg_metrikleri(y_gercek, y_tahmin):
    """Duyarlılık skoru metrikleri. MSE, 0/1 hedefte Brier skorudur; R² Brier beceri
    skorudur; AUC skorun sıralama başarısıdır (eşikten bağımsız)."""
    y_gercek = np.asarray(y_gercek, dtype=float)
    y_tahmin = np.asarray(y_tahmin, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_gercek, y_tahmin)),
        "MSE": float(mean_squared_error(y_gercek, y_tahmin)),
        "RMSE": float(np.sqrt(mean_squared_error(y_gercek, y_tahmin))),
        "R2": float(r2_score(y_gercek, y_tahmin)),
        "AUC": float(roc_auc_score(y_gercek, y_tahmin)),
    }


def cls_metrikleri(y_gercek, y_tahmin):
    y_gercek = np.asarray(y_gercek, dtype=object)
    y_tahmin = np.asarray(y_tahmin, dtype=object)
    return {
        "Accuracy": float(accuracy_score(y_gercek, y_tahmin)),
        "BalancedAcc": float(balanced_accuracy_score(y_gercek, y_tahmin)),
        "F1_macro": float(f1_score(y_gercek, y_tahmin, average="macro")),
        "F1_weighted": float(f1_score(y_gercek, y_tahmin, average="weighted")),
    }


def taban_cizgileri(df):
    """Dört taban çizgisi; "modeller gerçekten öğreniyor mu?" sorusunun ölçütüdür.

    (a) Ortalama tabanı : her test noktasına eğitim kümesinin yangın oranı atanır (R² = 0).
    (b) Çoğunluk tabanı : her test noktasına eğitimdeki en kalabalık sınıf atanır.
    (c) Tek kural tabanı: eğitim kümesinde öğrenilen TEK bir eşik sorusu (bir düzeyli
                          karar ağacı, "karar kütüğü"). Regresyonda kuralın iki yanındaki
                          yangın oranları, sınıflandırmada kuralın iki yanındaki çoğunluk
                          sınıfı tahmin edilir.
    (a) ve (b) hiçbir şey bilmeyen modeldir; (c) yangın duyarlılığı için akla gelen en
    basit kuraldır ve her yöntemin geçmesi beklenir.
    """
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = bolunme(df)
    cogunluk = yc_tr.value_counts().idxmax()
    yc_tahmin = pd.Categorical([cogunluk] * len(yc_te), categories=SINIF_ETIKETLERI)
    kr = DecisionTreeRegressor(max_depth=1, random_state=RANDOM_STATE).fit(X_tr, yr_tr)
    kc = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE).fit(X_tr, np.asarray(yc_tr, dtype=object))
    # Sabit skor bütün noktaları eşit sıralar; AUC bu durumda 0,5'tir (rastgele sıralama).
    ortalama = np.full(len(yr_te), float(yr_tr.mean()))
    return {
        "regresyon": reg_metrikleri(yr_te, ortalama),
        "siniflandirma": cls_metrikleri(yc_te, yc_tahmin),
        "kural_regresyon": reg_metrikleri(yr_te, kr.predict(X_te)),
        "kural_siniflandirma": cls_metrikleri(yc_te, kc.predict(X_te)),
        "kural": {"ozellik": OZELLIKLER[kr.tree_.feature[0]], "esik": float(kr.tree_.threshold[0]),
                  "sol_oran": float(kr.tree_.value[1].ravel()[0]),
                  "sag_oran": float(kr.tree_.value[2].ravel()[0])},
        "cogunluk_sinifi": str(cogunluk),
        "egitim_yangin_orani": float(yr_tr.mean()),
    }


def reg_grafikleri(y_gercek, y_tahmin, algoritma, dosya_onaki):
    """Duyarlılık skorunun sınıflara göre dağılımı ve güvenilirlik (kalibrasyon) eğrisi."""
    y_gercek = np.asarray(y_gercek, dtype=float)
    y_tahmin = np.asarray(y_tahmin, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    kutu = np.linspace(min(0.0, y_tahmin.min()), max(1.0, y_tahmin.max()), 31)
    ax.hist(y_tahmin[y_gercek == 0], bins=kutu, alpha=0.6, color="#4c72b0", label="Yangın Yok (gerçek)")
    ax.hist(y_tahmin[y_gercek == 1], bins=kutu, alpha=0.6, color="#c44e52", label="Yangın (gerçek)")
    ax.axvline(0.5, color="k", ls="--", lw=1.2, label="Eşik (0,5)")
    ax.set_xlabel("Tahmin edilen duyarlılık skoru")
    ax.set_ylabel("Nokta sayısı")
    ax.set_title(f"{algoritma}: Skor Dağılımı")
    ax.legend(loc="upper center")
    ax = axes[1]
    kirp = np.clip(y_tahmin, 0, 1)
    kenar = np.linspace(0, 1, 11)
    kutu_no = np.clip(np.digitize(kirp, kenar) - 1, 0, 9)
    ort_tahmin, ort_gercek, sayim = [], [], []
    for i in range(10):
        s = kutu_no == i
        if s.sum() >= 5:
            ort_tahmin.append(kirp[s].mean())
            ort_gercek.append(y_gercek[s].mean())
            sayim.append(int(s.sum()))
    ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="Mükemmel kalibrasyon")
    ax.plot(ort_tahmin, ort_gercek, "o-", color="darkorange", lw=1.8, label="Model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Ortalama tahmin edilen skor")
    ax.set_ylabel("Gerçek yangın oranı")
    ax.set_title(f"{algoritma}: Güvenilirlik Eğrisi")
    ax.legend(loc="upper left")
    fig.tight_layout()
    yol = os.path.join(SEKIL_DIR, f"{dosya_onaki}_regresyon.png")
    fig.savefig(yol, bbox_inches="tight")
    plt.close(fig)
    return yol


def karmasiklik_matrisi(y_gercek, y_tahmin):
    return confusion_matrix(np.asarray(y_gercek, dtype=object), np.asarray(y_tahmin, dtype=object),
                            labels=SINIF_ETIKETLERI)


def karmasiklik_grafigi(y_gercek, y_tahmin, algoritma, dosya_onaki):
    """Karmaşıklık (confusion) matrisi ısı haritası; matris JSON'a da yazılır."""
    cm = karmasiklik_matrisi(y_gercek, y_tahmin)
    n = len(SINIF_ETIKETLERI)
    print(f"{algoritma} — karmaşıklık matrisi (satır: gerçek, sütun: tahmin)")
    print("  sınıf sırası:", SINIF_ETIKETLERI)
    for i, et in enumerate(SINIF_ETIKETLERI):
        print(f"  {et:>10} | " + "  ".join(f"{int(v):5d}" for v in cm[i]))
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(SINIF_ACIKLAMA)
    ax.set_yticklabels(SINIF_ACIKLAMA)
    ax.set_xlabel("Tahmin edilen sınıf")
    ax.set_ylabel("Gerçek sınıf")
    ax.set_title(f"{algoritma}: Karmaşıklık Matrisi (Test)")
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    yol = os.path.join(SEKIL_DIR, f"{dosya_onaki}_siniflandirma.png")
    fig.savefig(yol, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(SONUC_DIR, f"_km_{dosya_onaki}.json"), "w", encoding="utf-8") as f:
        json.dump({"siralama": SINIF_ETIKETLERI, "matris": cm.tolist()}, f, ensure_ascii=False)
    return yol


def ozellik_onem_grafigi(onemler, algoritma, dosya_onaki, baslik="Özellik Önemleri"):
    """Ağaç tabanlı modeller ve katsayılar için yatay çubuk grafiği; değerler JSON'a da yazılır."""
    onemler = np.asarray(onemler, dtype=float)
    s = pd.Series(onemler, index=[OKUNUR[c] for c in OZELLIKLER]).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    s.plot.barh(ax=ax, color="teal", edgecolor="k")
    ax.set_title(f"{algoritma}: {baslik}")
    ax.set_xlabel("Önem / |Katsayı|")
    fig.tight_layout()
    yol = os.path.join(SEKIL_DIR, f"{dosya_onaki}_onem.png")
    fig.savefig(yol, bbox_inches="tight")
    plt.close(fig)
    pay = onemler / onemler.sum() if onemler.sum() > 0 else onemler
    with open(os.path.join(SONUC_DIR, f"_onem_{dosya_onaki}.json"), "w", encoding="utf-8") as f:
        json.dump({"ozellik": OZELLIKLER, "onem": onemler.tolist(), "pay": pay.tolist()},
                  f, ensure_ascii=False, indent=1)
    return yol


def sinif_kodlari(y):
    """XGBoost/LightGBM/CatBoost gibi sayısal etiket isteyen kütüphaneler için
    kategorik sınıf etiketlerini 0/1 kodlarına çevirir (0 = Yangın Yok, 1 = Yangın)."""
    return pd.Series(pd.Categorical(y, categories=SINIF_ETIKETLERI).codes, index=getattr(y, "index", None))


def koddan_sinifa(kodlar):
    """0/1 kodlarını orijinal sınıf etiketlerine geri çevirir."""
    return pd.Categorical.from_codes(np.asarray(kodlar, dtype=int).ravel(),
                                     categories=SINIF_ETIKETLERI)


def sonuc_kaydet(algoritma_adi, reg, cls, ek=None):
    """Karşılaştırma notebook'u için sonuçları JSON olarak kaydeder."""
    kayit = {"algoritma": algoritma_adi, "regresyon": reg, "siniflandirma": cls}
    if ek:
        kayit["ek"] = ek
    yol = os.path.join(SONUC_DIR, algoritma_adi.lower().replace(" ", "_") + ".json")
    with open(yol, "w", encoding="utf-8") as f:
        json.dump(kayit, f, ensure_ascii=False, indent=2)
    print(f"Sonuçlar kaydedildi: {yol}")
    return kayit


def sozluk_json_yaz(df, ek=None):
    """Veri setinin özetini JSON olarak kaydeder (kitap metni için tek kaynak)."""
    m = egitim_maskesi(df)
    sayim = df["Sinif"].value_counts().reindex(SINIF_ETIKETLERI)
    ozet = {
        "satir_sayisi": int(len(df)),
        "ozellik_sayisi": int(len(OZELLIKLER)),
        "grup_sayisi": int(df[GRUP].nunique()),
        "egitim_satir": int(m.sum()),
        "test_satir": int((~m).sum()),
        "yangin_orani": float(df[HEDEF].mean()),
        "sinif_dagilimi": {k: int(v) for k, v in sayim.items()},
        "sinif_orani": {k: float(v) for k, v in (sayim / len(df)).round(4).items()},
        "egitim_sinif": {k: int(v) for k, v in df.loc[m, "Sinif"].value_counts().reindex(SINIF_ETIKETLERI).items()},
        "test_sinif": {k: int(v) for k, v in df.loc[~m, "Sinif"].value_counts().reindex(SINIF_ETIKETLERI).items()},
        "ortanca_sinif": {c: {k: float(v) for k, v in df.groupby("Sinif", observed=True)[c].median().items()}
                          for c in SAYISAL},
        "arazi_yangin_orani": {k: float(v) for k, v in df.groupby("Arazi")[HEDEF].mean().items()},
        "arazi_sayim": {k: int(v) for k, v in df["Arazi"].value_counts().items()},
        "baki_yangin_orani": {k: float(v) for k, v in df.groupby("Baki")[HEDEF].mean().items()},
    }
    if ek:
        ozet.update(ek)
    yol = os.path.join(SONUC_DIR, "_veri_ozeti.json")
    with open(yol, "w", encoding="utf-8") as f:
        json.dump(ozet, f, ensure_ascii=False, indent=2)
    print(f"Veri özeti kaydedildi: {yol}")
    return ozet
