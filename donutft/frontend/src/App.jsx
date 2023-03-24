import { useEffect, useState } from "react";
import reactLogo from "./assets/react.svg";
import viteLogo from "/vite.svg";
import "./App.css";
import {
  Box,
  Button,
  Checkbox,
  Flex,
  HStack,
  Image,
  Img,
  Input,
  Stack,
  Textarea,
} from "@chakra-ui/react";
import { useRest } from "./hooks/urls";
import urlJoin from "url-join";
import { useObject } from "./hooks/state";
import axios from "axios";

const BASE = "http://localhost:8080";

function Mycheckbox({ value, onChange }) {
  console.log("checked", value);
  return (
    <Checkbox
      checked={value}
      onChange={(e) => onChange({ target: { value: e.target.checked } })}
    />
  );
}

const editors = {
  text: (props) => <Textarea rows="20" {...props} />,
  complete: Mycheckbox,
};
function EditPanel({ id }) {
  const state = useObject();
  const [data] = useRest(urlJoin(BASE, "metadata", id));
  useEffect(() => {
    if (data != null) {
      state.update(data);
    }
  }, [data]);

  return (
    <Stack p="0.5rem">
      {Object.entries(state).map(([k, v]) => {
        const Comp = editors[k] || Input;
        return (
          <HStack key={k}>
            <Box>{k}</Box>
            <Comp value={v} onChange={(e) => (state[k] = e.target.value)} />
          </HStack>
        );
      })}
      <Button
        onClick={(e) => {
          axios.put(urlJoin(BASE, "metadata", id), state).then((resp) => {
            console.log(resp);
          });
        }}
      >
        Save
      </Button>
    </Stack>
  );
}

function App() {
  const [docs] = useRest(urlJoin(BASE, "docs"));
  const state = useObject();
  return (
    <Flex w="100vw" h="100vh" p="1rem">
      <Stack w="20%">
        {docs != null &&
          docs.map((el, i) => (
            <Box
              fontWeight={state.image == el && "bold"}
              onClick={(e) => (state.image = el)}
              key={el}
            >
              {i}.{el}
            </Box>
          ))}
      </Stack>
      <Box w="50%">
        <Image src={urlJoin(BASE, "docs", state.image || "")} />
      </Box>
      <Box w="30%" bg="red">
        {state.image && <EditPanel id={state.image} />}
      </Box>
    </Flex>
  );
}

export default App;
