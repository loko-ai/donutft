import axios from "axios";
import { useEffect, useState } from "react";
import { io } from "socket.io-client";

export function useRest(url) {
  const [data, setData] = useState();
  const [error, setError] = useState();
  const [loading, setLoading] = useState();
  const [count, setCount] = useState(0);
  useEffect(() => {
    if (url) {
      setLoading(true);
      setError(null);
      //setData(null);
      axios
        .get(url)
        .then((resp) => {
          setData(resp.data);
        })
        .catch((err) => {
          setError(err);
        })
        .finally(() => setLoading(false));
    }
  }, [url, count]);

  return [data, error, loading, () => setCount(count + 1)];
}

export function useSocket(url, topic, config = { transports: ["websocket"] }) {
  const [message, setMessage] = useState(null);

  useEffect(() => {
    const client = io(url, config);
    console.log(topic);
    client.on(topic, (data) => {
      console.log(topic);
      setMessage(data);
    });

    return () => {
      if (client) client.disconnect();
    };
  }, [url, topic]);

  return message;
}
export function useSocketRest(url, resource, socketUrl) {
  const [data, setData] = useState();
  const [error, setError] = useState();
  const [loading, setLoading] = useState();
  const [count, setCount] = useState(0);
  const message = useSocket(socketUrl, resource);
  useEffect(() => {
    if (url) {
      setLoading(true);
      setError(null);
      //setData(null);
      axios
        .get(url)
        .then((resp) => {
          setData(resp.data);
        })
        .catch((err) => {
          setError(err);
        })
        .finally(() => setLoading(false));
    }
  }, [url, count]);
  useEffect(() => {
    setCount(count + 1);
  }, [message]);

  return [data, error, loading, () => setCount(count + 1)];
}
