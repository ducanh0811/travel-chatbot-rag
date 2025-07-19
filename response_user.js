const body={
  "recipient":{
    "id": `${$('Edit Fields').first().json.body.entry[0].messaging[0].sender.id}`
  },
  "messaging_type": "RESPONSE",
  "message":{
    "text":`${$input.first().json.result}`
  }
}
  return {
  data: body
}