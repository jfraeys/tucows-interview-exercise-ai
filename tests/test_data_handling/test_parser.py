from src.data_handling.parser import parse_file

TEST_PDF = "tests/data/domain_support_docs.pdf"
TEST_TXT = "tests/data/CS_FAQ.txt"
TEST_MD = "tests/data/domain_suspension_guidelines.md"


def test_txt_faq_parser():
    chunks = parse_file(TEST_TXT)

    # Basic checks
    assert len(chunks) > 0, "No chunks parsed from PDF"
    assert len(chunks) == 10, "Unexpected number of chunks parsed"

    # Check structure of first chunk
    first_chunk = chunks[0]
    assert "id" in first_chunk
    assert "text" in first_chunk
    assert "metadata" in first_chunk
    assert "source" in first_chunk["metadata"]
    assert "question" in first_chunk["metadata"]

    print("last_chunk", chunks[-1])
    last_chunk = chunks[-1]
    assert last_chunk["metadata"]["question"].startswith("Q10:")
    assert "Typically, standard renewal fees apply." in last_chunk["text"]
