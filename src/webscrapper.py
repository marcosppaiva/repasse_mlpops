import logging
from datetime import date

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from enums import Condition, ResidenceType

logging.basicConfig(
    level=logging.INFO,
    format="SCRAPPER_APP - %(asctime)s - %(levelname)s - %(message)s",
)


def get_regions():
    return [
        ("Aveiro", "1"),
        ("Beja", "2"),
        ("Braga", "3"),
        ("Bragança", "4"),
        ("Castelo Branco", "5"),
        ("Coimbra", "6"),
        ("Évora", "7"),
        ("Faro", "8"),
        ("Guarda", "9"),
        ("Ilha da Graciosa", "24"),
        ("Ilha da Madeira", "19"),
        ("Ilha das Flores", "28"),
        ("Ilha de Porto Santo", "20"),
        ("Ilha de Santa Maria", "21"),
        ("Ilha de São Jorge", "25"),
        ("Ilha de São Miguel", "22"),
        ("Ilha do Corvo", "29"),
        ("Ilha do Faial", "27"),
        ("Ilha do Pico", "26"),
        ("Ilha Terceira", "23"),
        ("Leiria", "10"),
        ("Lisboa", "11"),
        ("Portalegre", "12"),
        ("Porto", "13"),
        ("Santarém", "14"),
        ("Setúbal", "15"),
        ("Viana do Castelo", "16"),
        ("Vila Real", "17"),
        ("Viseu", "18"),
    ]


def request_page(url):
    request = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        timeout=40,
    )
    soup = BeautifulSoup(request.content, "html.parser")

    return soup


def create_request_link(residence_type, region, page, service_type="arrendar"):
    space = " "
    dash = "-"
    return (
        f"https://www.imovirtual.com/{service_type}/{residence_type}"
        + f"/{region[0].lower().replace(space,dash)}"
        + f"/?search%5Bregion_id%5D={region[1]}&nrAdsPerPage=72&page={page}"
    )


def get_num_pages(soup):
    try:
        page_list = soup.find("ul", attrs={"class": "pager"})
        total_pages = page_list.find_all("li")[-2].text
    except (AttributeError, TypeError):
        total_pages = 1
    return int(total_pages)


def get_attribute_safe(element, name, attribute, default="NA"):
    try:
        found_element = element.find(name, attribute)
        return found_element["alt"] if name == "img" else found_element.text
    except (AttributeError, TypeError):
        return default


def get_infos(soup):
    list_ads = soup.find_all("article")
    ads = []
    for ad in list_ads:
        price = (
            get_attribute_safe(ad, "li", {"class": "offer-item-price"})
            .strip()
            .split("€")[0]
        )
        energy_certify = get_attribute_safe(ad, "div", {"class": "energy-certify"})
        metric = get_attribute_safe(ad, "strong", {"class": "visible-xs-block"})
        description = get_attribute_safe(ad, "span", {"class": "offer-item-title"})
        location = get_attribute_safe(ad, "p", {"class": "text-nowrap"})
        rooms = get_attribute_safe(ad, "li", {"class": "offer-item-rooms hidden-xs"})
        details = get_attribute_safe(
            ad, "ul", {"class": "params-small clearfix hidden-xs"}
        )
        company = get_attribute_safe(ad, "img", {"company-logo lazy"})

        ads.append(
            {
                "price": price.strip(),
                "energy_certify": energy_certify.strip(),
                "metric": metric.strip(),
                "description": description.strip(),
                "location": location.strip().split(":")[1].strip(),
                "rooms": rooms.strip(),
                "details": details.strip(),
                "company": company.strip(),
            }
        )

    return ads


def detail_extract(data_frame: pd.DataFrame):
    data_frame.loc[
        data_frame[data_frame.details.str.contains("Anúncio")].index, "company"
    ] = "Anúncio Particular"
    data_frame["bathroom"] = (
        data_frame.details.astype("str")
        .str.extractall("(\d+)")
        .unstack()
        .fillna("")
        .sum(axis=1)
        .astype(int)
    )
    data_frame["condition"] = data_frame["details"].str.extract(
        f"({'|'.join([condition.value for condition in Condition])})"
    )
    data_frame = data_frame.drop(["details"], axis=1)

    return data_frame


def save_data(df: pd.DataFrame, on_cloud: bool = True):
    filename = f"data/raw/imovirtual.parquet"
    df.to_parquet(filename)
    # if on_cloud:
    #     try:
    #         bucket_utils.upload_file(filename, filename)
    #     except ParamValidationError as error:
    #         logging.error(f"Error to save file on cloud: {error}")


def scrapper_run():
    df = pd.DataFrame()
    regions = get_regions()
    for residence in ResidenceType:
        for region in tqdm(regions, desc=f"{residence}"):
            url = create_request_link(
                residence_type=residence.name.lower(), region=region, page="1"
            )
            soup = request_page(url)
            pages = get_num_pages(soup)

            for page in range(1, pages + 1):
                url = create_request_link(
                    residence_type=residence.name.lower(), region=region, page=page
                )
                soup = request_page(url)
                ads = get_infos(soup)

                temp = pd.DataFrame(ads)
                temp["property_type"] = residence.name.lower()
                temp["district"] = region[0]
                df = pd.concat([df, temp])

    df["extract_date"] = date.today()
    df["extract_date"] = pd.to_datetime(df["extract_date"])
    df = df.reset_index(drop=True)

    df = detail_extract(df)

    save_data(df)


if __name__ == "__main__":
    scrapper_run()
