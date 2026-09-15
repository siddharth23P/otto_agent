"""Element ids and swipe start points. A web page's form buttons can all read
"Submit" (Amazon's product page on a Galaxy S23, 2026-09-15: nineteen of
them, Add to Cart and Buy Now among them); they are told apart, and judged,
by their resource id. A swipe starts where it is asked to."""
from agent.phone import JsonBackend, digest, guard, phone_tools
from tests.phone_fakes import AMAZON_PRODUCT, BLINKIT_SEARCH, FakePhone, node, snapshot


def _tools(phone):
    return {t.name: t.call for t in phone_tools(JsonBackend(phone))}


def test_an_id_reads_as_words_whatever_its_spelling():
    assert guard.id_words("buy-now-button") == "buy now button"
    assert guard.id_words("in.amazon.mShop.android.shopping:id/buyNowButton") == "buy now button"
    assert guard.id_words("buybox.addToCart") == "buybox add to cart"
    assert guard.id_words("") == ""


def test_the_id_is_judged_with_the_label_and_the_stricter_verdict_wins():
    assert guard.target_verdict("Submit", view_id="buy-now-button") == "pay"
    assert guard.target_verdict("", view_id="com.shop:id/buyNow") == "pay"
    assert guard.target_verdict("Submit", view_id="add-to-cart-button") == "commit"  # the label's own verdict
    assert guard.target_verdict("Add to cart", view_id="add-to-cart-button") == ""
    assert guard.target_verdict("Pay now", view_id="add-to-cart-button") == "pay"
    assert guard.target_verdict("", ["Total ₹28"], view_id="continue-button") == "pay"
    assert guard.target_verdict("Buyer's guide", view_id="buyers-guide") == ""  # whole words, as for a label


def test_enter_is_refused_on_a_screen_whose_pay_button_says_so_only_in_its_id():
    texts = [digest.label_of(n) for n in AMAZON_PRODUCT["nodes"]]
    ids = [digest.view_id_of(n) for n in AMAZON_PRODUCT["nodes"] if n.get("v")]
    assert guard.submit_verdict(texts) == ""
    assert "buy-now-button" in guard.submit_verdict(texts, ids)


def test_the_digest_shows_an_actionable_elements_id_and_only_that():
    text = digest.render_digest(AMAZON_PRODUCT)
    assert '[5] "Submit" button clickable #add-to-cart-button @719,2070' in text
    assert '[6] "Submit" button clickable #buy-now-button @719,2250' in text
    plain = snapshot("p", "a.b", "Shop", [node(1, "₹559", v="price-block")])
    assert "#" not in digest.render_digest(plain)


def test_a_target_may_name_the_id_or_its_words_and_a_label_still_outranks_it():
    assert digest.find_node(AMAZON_PRODUCT, "Add to cart") == (5, [])
    assert digest.find_node(AMAZON_PRODUCT, "#add-to-cart-button") == (5, [])
    assert digest.find_node(AMAZON_PRODUCT, "buy-now-button") == (6, [])
    assert digest.find_node(AMAZON_PRODUCT, "Submit") == (None, [2, 3, 5, 6])
    shop = snapshot("c", "a.b", "Shop", [node(1, "Cart", c=True), node(2, "", c=True, v="cart")])
    assert digest.find_node(shop, "cart") == (1, [])
    assert digest.find_node(shop, "#cart") == (2, [])


def test_add_to_cart_is_reached_by_its_id_and_buy_now_is_refused_whatever_it_is_called():
    phone = FakePhone([AMAZON_PRODUCT] * 8)
    by_name = _tools(phone)
    assert by_name["phone_screen"]("{}").ok
    result = by_name["phone_commit"]('{"target": "Submit"}')
    assert not result.ok and "#add-to-cart-button" in result.stderr and "#buy-now-button" in result.stderr
    result = by_name["phone_act"]('{"op": "tap", "target": "add to cart"}')
    assert not result.ok and "phone_commit" in result.stderr  # its label is still "Submit"
    result = by_name["phone_commit"]('{"target": "#add-to-cart-button"}')
    assert result.ok and phone.calls[-1] == ("tap_node", "s6", 5, False, True)
    for target in ("buy now", "#buy-now-button"):
        result = by_name["phone_commit"](f'{{"target": "{target}"}}')
        assert result.stderr.startswith("GUARD:") and "'Submit' #buy-now-button is a payment step" in result.stderr, target
    result = by_name["phone_act"]('{"op": "press", "key": "enter"}')
    # Not a checkout, so not a hand-over: Enter is declined and the model is
    # pointed at the button it means.
    assert not result.ok and not result.stderr.startswith("GUARD:")
    assert "buy-now-button" in result.stderr and "Add to Cart" in result.stderr
    assert not any(c[0] == "tap_node" and c[2] == 6 for c in phone.calls)
    assert not any(c[0] == "press" for c in phone.calls)


def test_enter_is_judged_on_the_screen_as_it_is_now_not_as_it_was_last_read():
    """2026-09-15, a Galaxy S23: a tapped Amazon search suggestion returned
    before the results loaded, Enter was pressed on results carrying "Buy for
    ₹71,549 with HDFC", and the phone's guard handed the whole run over."""
    suggestions = snapshot("sg", "in.amazon.mShop.android.shopping", "Amazon", [
        node(1, "samsung galaxy z flip6 5g 256gb", r="view", b=(0, 400, 1440, 520), c=True),
    ])
    results = snapshot("rs", "in.amazon.mShop.android.shopping", "Amazon", [
        node(1, "Samsung Galaxy Z Flip6 5G (256GB)", r="view", b=(0, 900, 1440, 1000), c=True),
        node(2, "₹71,599 M.R.P: ₹1,09,999 (35% off) Buy for ₹71,549 with HDFC Bank credit card", r="view",
             b=(0, 1000, 1440, 1100), c=True),
    ])
    phone = FakePhone([suggestions, suggestions, results, results])
    by_name = _tools(phone)
    assert by_name["phone_screen"]("{}").ok
    assert by_name["phone_act"]('{"op": "tap", "target": "samsung galaxy z flip6 5g 256gb"}').ok
    result = by_name["phone_act"]('{"op": "press", "key": "enter"}')
    assert not result.ok and not result.stderr.startswith("GUARD:")
    assert "Buy for ₹71,549" in result.stderr and "Add to Cart" in result.stderr
    assert not any(call[0] == "press" for call in phone.calls)


def test_a_swipe_starts_where_asked_and_is_judged_by_what_is_under_it():
    slider = snapshot("sl", "com.example.shop", "Shop", [
        node(1, "Photos", r="list", b=(0, 300, 1080, 900), s=True),
        node(2, "Slide to pay ₹499", r="view", b=(60, 2100, 1020, 2250), c=True),
    ])
    phone = FakePhone([slider] * 4)
    by_name = _tools(phone)
    assert by_name["phone_screen"]("{}").ok
    assert by_name["phone_act"]('{"op": "swipe", "direction": "left", "x": 900, "y": 600}').ok
    assert phone.calls[-1] == ("swipe", "left", 900, 600)
    result = by_name["phone_act"]('{"op": "swipe", "direction": "right", "x": 100, "y": 2170}')
    assert result.stderr.startswith("GUARD:") and "payment step" in result.stderr
    assert by_name["phone_act"]('{"op": "swipe", "direction": "up"}').ok
    assert phone.calls[-1] == ("swipe", "up")
    assert sum(1 for c in phone.calls if c[0] == "swipe") == 2


def test_the_serve_proxy_passes_a_swipe_start_point_through():
    """`otto serve`'s SocketPhone took (direction) only, so every swipe from a
    point failed over the socket as a TypeError."""
    from agent.server.proxy import SocketPhone

    phone = SocketPhone.__new__(SocketPhone)
    seen = []
    phone._call = lambda method, *args: seen.append((method, *args)) or {"done": "swiped"}
    assert JsonBackend(phone).swipe("up") == {"done": "swiped"}
    assert JsonBackend(phone).swipe("left", 900, 600) == {"done": "swiped"}
    assert seen == [("swipe", "up"), ("swipe", "left", 900, 600)]


def test_the_swipe_start_point_is_sent_only_when_there_is_one():
    phone = FakePhone([BLINKIT_SEARCH] * 2)
    backend = JsonBackend(phone)
    backend.swipe("up")
    backend.swipe("left", 900, 600)
    assert phone.calls == [("swipe", "up"), ("swipe", "left", 900, 600)]

    class OldApp:
        """An app built before the start point existed: a one-argument swipe."""

        def swipe(self, direction):
            return '{"ok": true, "data": {"done": "swiped"}}'

    assert JsonBackend(OldApp()).swipe("down") == {"done": "swiped"}
