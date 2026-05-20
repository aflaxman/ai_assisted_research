"""
Known Gavi HSS document URLs and metadata.

This registry maps countries to their HSS proposal document URLs
found on gavi.org. Each entry represents one document (proposal/summary).
"""

# Known HSS proposal document URLs from gavi.org
# Format: list of dicts with country, url, doc_type, year (if known), format
DOCUMENT_SOURCES = [
    # Ethiopia - older HSS proposal (DOC format)
    {
        "country": "Ethiopia",
        "iso3": "ETH",
        "url": "https://www.gavi.org/sites/default/files/document/proposal-for-hss-support--ethiopia(2)doc.doc",
        "doc_type": "approved_proposal",
        "year": None,  # Year not in URL; determine from document content
        "format": "doc",
        "gavi_country_docs_url": "https://www.gavi.org/country-documents/ethiopia",
    },
    # Nigeria - older HSS proposal (DOC format)
    {
        "country": "Nigeria",
        "iso3": "NGA",
        "url": "https://www.gavi.org/sites/default/files/document/proposal-for-hss-support--nigeriadoc.doc",
        "doc_type": "approved_proposal",
        "year": None,
        "format": "doc",
        "gavi_country_docs_url": "https://www.gavi.org/country-documents/nigeria",
    },
    # DRC - older HSS proposal (DOC format)
    {
        "country": "Democratic Republic of the Congo",
        "iso3": "COD",
        "url": "https://www.gavi.org/sites/default/files/document/proposal-for-hss-support--congo,-democratic-republic-of-thedoc.doc",
        "doc_type": "approved_proposal",
        "year": None,
        "format": "doc",
        "gavi_country_docs_url": "https://www.gavi.org/country-documents/democratic-republic-of-the-congo",
    },
    # Malawi - older HSS proposal (DOC format)
    {
        "country": "Malawi",
        "iso3": "MWI",
        "url": "https://www.gavi.org/sites/default/files/document/proposal-for-hss-support--malawidoc.doc",
        "doc_type": "approved_proposal",
        "year": None,
        "format": "doc",
        "gavi_country_docs_url": "https://www.gavi.org/country-documents/malawi",
    },
    # Kenya - HSS proposal page (need to resolve to direct download)
    {
        "country": "Kenya",
        "iso3": "KEN",
        "url": "https://www.gavi.org/sites/default/files/document/proposal-for-hss-support--kenyadoc.doc",
        "doc_type": "approved_proposal",
        "year": None,
        "format": "doc",
        "gavi_country_docs_url": "https://www.gavi.org/country-documents/kenya",
    },
    # Pakistan - older HSS proposal (DOC format)
    {
        "country": "Pakistan",
        "iso3": "PAK",
        "url": "https://www.gavi.org/sites/default/files/document/proposal-for-hss-support--pakistandoc.doc",
        "doc_type": "approved_proposal",
        "year": None,
        "format": "doc",
        "gavi_country_docs_url": "https://www.gavi.org/country-documents/pakistan",
    },
    # Chad - older HSS proposal (DOC format)
    {
        "country": "Chad",
        "iso3": "TCD",
        "url": "https://www.gavi.org/sites/default/files/document/proposal-for-hss-support--chaddoc.doc",
        "doc_type": "approved_proposal",
        "year": None,
        "format": "doc",
        "gavi_country_docs_url": "https://www.gavi.org/country-documents/chad",
    },
    # India - 2017 HSS proposal (PDF format)
    {
        "country": "India",
        "iso3": "IND",
        "url": "https://www.gavi.org/sites/default/files/document/proposal-for-hss-support-2017--indiapdf.pdf",
        "doc_type": "approved_proposal",
        "year": 2017,
        "format": "pdf",
        "gavi_country_docs_url": "https://www.gavi.org/country-documents/india",
    },
    # Lao PDR - older HSS proposal (DOC format)
    {
        "country": "Lao PDR",
        "iso3": "LAO",
        "url": "https://www.gavi.org/sites/default/files/document/proposal-for-hss-support--lao-people-s-democratic-republicdoc.doc",
        "doc_type": "approved_proposal",
        "year": None,
        "format": "doc",
        "gavi_country_docs_url": "https://www.gavi.org/country-documents/lao-peoples-democratic-republic",
    },
    # Togo - older HSS proposal (DOC format)
    {
        "country": "Togo",
        "iso3": "TGO",
        "url": "https://www.gavi.org/sites/default/files/document/proposal-for-hss-support--togodoc.doc",
        "doc_type": "approved_proposal",
        "year": None,
        "format": "doc",
        "gavi_country_docs_url": "https://www.gavi.org/country-documents/togo",
    },
    # Central African Republic - 2017-2019 HSS proposal (PDF format)
    {
        "country": "Central African Republic",
        "iso3": "CAF",
        "url": "https://www.gavi.org/sites/default/files/document/2020/Proposal%20for%20HSS%20support-%20(2017-2019)%20CAR.pdf",
        "doc_type": "approved_proposal",
        "year": 2017,
        "format": "pdf",
        "gavi_country_docs_url": "https://www.gavi.org/country-documents/central-african-republic",
    },
    # Uzbekistan - HSS proposal (PDF format)
    {
        "country": "Uzbekistan",
        "iso3": "UZB",
        "url": "https://www.gavi.org/sites/default/files/document/proposal-for-hss-support--uzbekistanpdf.pdf",
        "doc_type": "approved_proposal",
        "year": None,
        "format": "pdf",
        "gavi_country_docs_url": "https://www.gavi.org/country-documents/uzbekistan",
    },
]

# Additional URLs found for second time-point documents
ADDITIONAL_SOURCES = [
    # Cambodia - HSS evaluation (not a proposal, but useful backup)
    {
        "country": "Cambodia",
        "iso3": "KHM",
        "url": "https://www.gavi.org/sites/default/files/document/hss-evaluation-cambodiapdf.pdf",
        "doc_type": "evaluation",
        "year": None,
        "format": "pdf",
        "gavi_country_docs_url": "https://www.gavi.org/country-documents/cambodia",
    },
    # Somalia - older HSS proposal (DOC format)
    {
        "country": "Somalia",
        "iso3": "SOM",
        "url": "https://www.gavi.org/sites/default/files/document/proposal-for-hss-support--somaliadoc.doc",
        "doc_type": "approved_proposal",
        "year": None,
        "format": "doc",
        "gavi_country_docs_url": "https://www.gavi.org/country-documents/somalia",
    },
]

if __name__ == "__main__":
    print(f"Primary documents: {len(DOCUMENT_SOURCES)}")
    for d in DOCUMENT_SOURCES:
        print(f"  {d['country']} ({d['format'].upper()}): {d['url'][:80]}...")
