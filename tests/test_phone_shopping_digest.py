"""A shopping app's results page as the model reads it. Amazon's search
results (a Galaxy S23, 2026-09-15) hand each product to accessibility three
times, a price as a summary plus its pieces, the first screen entirely to
sponsored placements, and one listing a tracking query as its label. The
model quoted the second ad as "the cheapest iPhone"."""
from agent.phone import PHONE_GUIDANCE, JsonBackend, digest, phone_tools
from tests.phone_fakes import FakePhone, node, snapshot

RESULTS = snapshot("r1", "in.amazon.mShop.android.shopping", "Amazon", [
    node(1, "", d="Search", r="button", b=(165, 162, 1440, 312), c=True, v="chrome_search_box"),
    node(2, "Search", r="button", b=(250, 200, 336, 290)),
    node(3, "apple iphone", r="text", b=(360, 200, 952, 290)),
    node(4, "", d="All Filters Icon", r="button", b=(135, 560, 228, 650), c=True, v="s-all-filters-announce"),
    node(5, "", d="ref=sr_1_1_sspa?ie=UTF8&psc=1&spc=MTox", r="view", b=(15, 977, 607, 2013), c=True),
    node(6, "", d="View Sponsored information or leave ad feedback", r="view", b=(618, 923, 1383, 1036), c=True),
    node(7, "", d="Sponsored Ad - iPhone 16e 128 GB: Built for Apple Intelligence", r="view",
         b=(618, 1107, 1383, 1261), c=True),
    node(8, "", d="Sponsored Ad - iPhone 16e 128 GB: Built for Apple Intelligence", r="view",
         b=(618, 1107, 1383, 1261), c=True),
    node(9, "", d="Sponsored Ad - iPhone 16e 128 GB: Built for Apple Intelligence", r="text",
         b=(618, 1107, 1383, 1261)),
    node(10, "", d="₹59,900 M.R.P: ₹69,900 (14% off) Buy for ₹59,850 with HDFC Bank credit card", r="view",
         b=(618, 1407, 1383, 1733), c=True),
    node(11, "₹59,900", r="text", b=(618, 1411, 933, 1516)),
    node(12, "Buy for ₹59,850", r="text", b=(641, 1613, 975, 1673)),
    node(13, "Add to cart", r="button", b=(618, 1950, 1383, 2060), c=True),
    node(14, "iPhone 15 128 GB", r="view", b=(618, 2300, 1383, 2400), c=True),
    node(15, "", d="Home Tab 1 of 6", r="view", b=(0, 2698, 240, 2908)),
    node(16, "Home", r="text", b=(60, 2820, 180, 2885)),
    node(17, "Apple", r="checkbox", b=(618, 2450, 900, 2520), c=True, k=False),
    node(18, "Apple", r="text", b=(640, 2460, 880, 2510)),
])


def test_a_product_is_one_line_and_a_price_is_not_read_out_piece_by_piece():
    text = digest.render_digest(RESULTS)
    assert text.count("iPhone 16e 128 GB") == 1
    for folded in ("[2]", "[8]", "[9]", "[11]", "[12]", "[16]", "[18]"):
        assert folded not in text, folded
    assert '[10] "₹59,900 M.R.P: ₹69,900 (14% off)' in text
    assert '[3] "apple iphone" text' in text                     # the query is not the search box's label
    assert '[15] "Home Tab 1 of 6" view' in text
    assert '[13] "Add to cart" button clickable' in text
    assert '[17] "Apple" checkbox clickable unchecked' in text   # a checkable node keeps its line


def test_sponsored_results_are_marked_and_tracking_links_are_not_read_out():
    text = digest.render_digest(RESULTS)
    assert '[7] "iPhone 16e 128 GB: Built for Apple Intelligence" view clickable ad @1000,1184' in text
    assert '[14] "iPhone 15 128 GB" view clickable @1000,2350' in text  # an organic result carries no mark
    assert '[6] "View Sponsored information or leave ad feedback" view clickable ad @' in text
    assert "[5] (link) view clickable @311,1495" in text and "sr_1_1" not in text


def test_a_folded_node_is_still_a_target_by_its_text():
    assert digest.find_node(RESULTS, "₹59,900") == (11, [])
    assert digest.find_node(RESULTS, "iPhone 16e") == (None, [7, 8, 9])


def test_an_element_with_no_text_is_named_by_its_number():
    assert digest.find_node(RESULTS, "[5]") == (5, [])
    assert digest.find_node(RESULTS, "[99]") == (None, [])
    assert digest.find_node(RESULTS, "[15]", clickable_only=True) == (None, [])
    locked = snapshot("pw", "a.b", "Notes", [node(1, "", r="edit-field", e=True, p=True)])
    assert digest.find_node(locked, "[1]") == (None, [])


def test_one_list_among_several_is_scrolled_by_its_number():
    """Amazon's filter panel: the page's WebView behind, the category list
    (no text) in front, Sort by near the end of it."""
    panel = snapshot("fp", "in.amazon.mShop.android.shopping", "Amazon", [
        node(1, "", r="web", b=(0, 350, 1440, 2698), s=True),
        node(2, "", r="view", b=(0, 520, 453, 2858), s=True),
        node(3, "GenAI Model", r="view", b=(0, 2276, 453, 2475), c=True),
    ])
    phone = FakePhone([panel] * 3)
    by_name = {t.name: t.call for t in phone_tools(JsonBackend(phone))}
    assert by_name["phone_screen"]("{}").ok
    assert by_name["phone_act"]('{"op": "scroll", "direction": "down", "target": "[2]"}').ok
    assert phone.calls[-1] == ("scroll", "down", 2)


def test_the_guidance_says_how_to_find_the_cheapest():
    assert "marked ad" in PHONE_GUIDANCE and "sort and filters" in PHONE_GUIDANCE
    assert "never from memory" in PHONE_GUIDANCE
