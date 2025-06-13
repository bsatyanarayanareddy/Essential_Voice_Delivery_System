function startOrder() {
  fetch('/start-order', { method: 'POST' })
    .then(res => res.json())
    .then(data => alert(data.message))
    .catch(err => alert("Error: " + err));
}
